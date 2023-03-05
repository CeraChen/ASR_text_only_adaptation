import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


from espnet2.asr.adapter.conformer_adapter import ConformerAdapter  #here

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield





import numpy as np

def generation_token_sequence(
    batch_tokens: torch.Tensor,
    token_lengths: torch.LongTensor,
    pb: np.array,
    ps: np.array,
    blankid: int,
    ):
    """
    batch_tokens: _I _LI KE _MOV IES
    unpad_batch_tokens: <b> <b> <b> _I _I <b> <b> <b> _LI KE <b> _MOV _MOV <b><b> IES
    pb/ps: repeat_times: prob
    rp: [0   ,   1,   2,   3, ...]
    pb: [0.01, 0.2, 0.3, 0.1, ...]
    ps: [0.01, 0.1, 0.2, 0.3, ...]
    blankid: need to be verified

    Return:
    - new_batch_tokens: unpad tokens
    - new_batch_tokens_lengths: actually it is useless, as we can squeeze to get the result
    """
    assert batch_tokens.ndim == 2 and token_lengths.ndim == 1
    REPEAT_ARRAY = np.arange(0,1000)
    bs = batch_tokens.shape[0]
    #fb = np.cumsum(pb)
    #fs = np.cumsum(ps) # accum prob, encourage the longer repeatations? sounds bad

    new_batch_tokens = []
    ilens = []
    new_batch_tokens_lengths = []
    for bi in range(bs):
        n1 = np.random.choice(a=REPEAT_ARRAY[:pb.shape[0]], size=int(token_lengths[bi]+1), replace=True, p=pb) # #tokens + #blank (which is #tokens + 1)
        n2 = np.random.choice(a=REPEAT_ARRAY[:ps.shape[0]], size=int(token_lengths[bi]+1), replace=True, p=ps)
        appear_times = np.ndarray.flatten(np.concatenate((n1, n2), axis=0), "F")[:token_lengths[bi]*2+1]
        # token_lengths[bi]*2+1
       
        #logging.warning("repeat_times.shape: "+str(repeat_times.shape)) 
        tokens = batch_tokens[bi][:token_lengths[bi]]


        ctc_tokens = np.array([])
        for i, token in enumerate(tokens):
            cur_n1 = appear_times[2*i]
            cur_n2 = appear_times[2*i+1]
            while i != 0 and token == tokens[i-1] and cur_n1 == 0:
                cur_n1 = np.random.choice(a=REPEAT_ARRAY[:pb.shape[0]], size=1, replace=True, p=pb)
            ctc_tokens = np.append(ctc_tokens, np.repeat(blankid, cur_n1))
            ctc_tokens = np.append(ctc_tokens, np.repeat(token.cpu().numpy(), cur_n2))
        ctc_tokens = np.append(ctc_tokens, np.repeat(blankid, n1[-1]))
        #logging.warning("CTC Token:" + str(ctc_tokens))
        ctc_tokens = torch.from_numpy(ctc_tokens).to(batch_tokens.device)
        ctc_tokens = torch.tensor(ctc_tokens, dtype=torch.float32)
        ilens.append(len(ctc_tokens))
        #logging.warning("ilens: "+str(ilens))
        new_batch_tokens.append(ctc_tokens)


    masks = (~make_pad_mask(ilens)[:, None, :]).to(new_batch_tokens[0].device)
    return torch.nn.utils.rnn.pad_sequence(new_batch_tokens, batch_first=True, padding_value=-1), masks










class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        adapter: ConformerAdapter, #here
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        use_adapter: bool = False, #here
        training_step: int = 0,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        
        self.adapter = adapter #here
        self.use_adapter = use_adapter
        self.training_step = training_step
        logging.warning("If an adapter is added, training step is{}".format(self.training_step))
        
        if self.use_adapter:#here
            assert self.training_step > 0, self.training_step < 4

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        #target_speech: torch.Tensor = None,
        #target_speech_lengths: torch.Tensor = None,
        target_text: torch.Tensor = None,
        target_text_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        
        

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            #encoder_out = encoder_out[0]
            if self.use_adapter:
                masks_outs = encoder_out[2]
                #logging.warning("after encode, masks[0][0].shape:{}".format(masks_outs[0][0].shape))
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()
       

        count_blank = torch.zeros(1000, dtype=torch.long)
        count_none_blank = torch.zeros(1000, dtype=torch.long) 
                
        alpha = 0.0
        if self.use_adapter:#here
            if self.training_step == 2:
                # 以intermediate[1]为label
                # intermediate[0] argmax转CTC sequence
                # 以此作为输入送进adapter
                # 输出与label算mseloss
                
                #logging.warning("Training step 2 with an adapter in espnet forward")
               
                input_intermediate_out = intermediate_outs[0][1]
                input_intermediate_out = self.ctc.softmax(input_intermediate_out)                            
                #logging.warning("input_intermediate_out.shape:{}".format(input_intermediate_out.shape))
                x = torch.argmax(input_intermediate_out, -1, True).to(torch.float32) 
                #x = torch.argmax(intermediate_outs[0][1], -1, True).to(torch.float32) #咋整
                #logging.warning("before adapter, x.shape:{}".format(x.shape))
                #logging.warning("before adapter, x_masks.shape:{}".format(masks_outs[0][0].shape))
                #logging.warning("before adapter, x_masks:{}".format(masks_outs[0][0]))
               




                #for i in range(int(x.shape[0])):
                #    sequence = x[i]
                #    prev_token = -1
                #    cur_count = 0
                #    for cur_token in sequence:
                #        cur_count += 1
                #        if cur_token == prev_token:
                #            #cur_count += 1
                #            continue
                #        if prev_token == 0:
                #            count_blank[cur_count] += 1
                #        else:
                #            count_none_blank[cur_count] += 1
                #            if cur_token != 0:
                #                count_blank[0] += 1
                #        cur_count = 0
                #        prev_token = cur_token 

                y, _ = self.adapter(x, masks_outs[0][0])
                #logging.warning("encoder_out_lens:{}".format(encoder_out_lens))
                #y = self.adapter(x, encoder_out_lens)
                loss_fn = torch.nn.MSELoss()
                #logging.warning("y.shape: {}".format(y.shape))
                #logging.warning("label.shape: {}".format(intermediate_outs[1][1].shape))
                #logging.warning("y[0]: {}".format(y[0]))
                #logging.warning("intermediate_outs[1][1][0]: {}".format(intermediate_outs[1][1][0]))
                loss_adpt = loss_fn(y, intermediate_outs[1][1])
                #loss_adpt = torch.nn.MSELoss(y, intermediate_outs[1][1])
                #logging.warning("Adapter MSE loss:{}".format(loss_adpt))
                stats["loss"] = loss_adpt.detach()
                #stats["loss"] = (loss_adpt.detach(), (count_blank, count_none_blank))

                #stats["count"] = (count_blank, count_none_blank)


                # logging.warning("curren_blank: {}".format(count_blank))
                # logging.warning("current_none_blank: {}".format(count_none_blank))
                # force_gatherable: to-device and to-tensor if scalar for DataParallel
                loss_adpt, stats, weight = force_gatherable((loss_adpt, stats, batch_size), loss_adpt.device)
                return loss_adpt, stats, weight
            
            # if self.training_step == 3:
            #     # target_text转CTC sequence
            #     x = None
            #     y = self.adapter(x)
            #     encoder_out = 0.5*encoder_out + 0.5*y               
                
        
            if self.training_step == 3 and self.training:
                # target_text转CTC sequence
                import numpy as np
                count_blank = np.array([1975726, 100541, 11602, 1854, 378, 113, 27, 9, 1, 2, 2, ], dtype=float)
                pb = count_blank/np.sum(count_blank)
                #logging.warning(str(pb))
                #pb[-1] = 1-np.sum(pb[:-1])
                #logging.warning(str(pb))
                
                count_none_blank = np.array([0, 1854816, 190362, 35587, 8562, 2327, 648, 197, 56, 29, 9, 2, 3, ], dtype=float)
                ps = count_none_blank/np.sum(count_none_blank)
                #logging.warning(str(ps))
                #ps[-1] = 1-np.sum(ps[:-1])
                #logging.warning("target_text_lengths.shape: "+str(target_text_lengths.shape))
                #logging.warning("target_text_lengths: "+str(target_text_lengths))
                #logging.warning("text.shape: "+str(text.shape))
                #logging.warning("text: "+str(text))
                
                x, x_lens = generation_token_sequence(target_text, target_text_lengths, pb, ps, 0)
                # #logging.warning(str(x_lens.shape))
                #x = self.encoder.embed(x.unsqueeze(-1))
                y, y_lens = self.adapter(x.unsqueeze(-1), x_lens)#(x, x_lens)
                output, output_lens, _ = self.encoder(y, y_lens, only_higher_layers = True)
                loss_target, _ = self._calc_ctc_loss(output, output_lens, target_text, target_text_lengths)
                stats["loss_target"] = loss_target.detach()
                #logging.warning(str(loss_target))
                alpha = 0.5
                
        
        

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            if self.use_adapter and self.training_step == 3 and self.training:
            	loss_ctc = alpha * ((
                	1 - self.interctc_weight
            	) * loss_ctc + self.interctc_weight * loss_interctc) + (1-alpha) * loss_target
            else:
                loss_ctc = (1 - self.interctc_weight) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        masks_outs = None
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc, masks_out=self.use_adapter
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths, masks_out=self.use_adapter)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            if self.use_adapter:
                masks_outs = encoder_out[2]
                #logging.warning("after encoder, masks_outs[0][0].shape".format(masks_outs[0][0].shape))
            intermediate_outs = encoder_out[1]
            #logging.warning("encoder_out[2][0][0].shape:{}".format(encoder_out[2][0][0].shape))
            #logging.warning("encoder_out.len:{}".format(len(encoder_out)))
            encoder_out = encoder_out[0]
            

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            if self.use_adapter:
                return (encoder_out, intermediate_outs, masks_outs), encoder_out_lens  
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

