import logging
import torch
from torch.nn.modules import Linear, Dropout, Dropout2d
from allennlp.modules import Seq2SeqEncoder, TimeDistributed
from allennlp.common.checks import check_dimensions_match

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Seq2SeqEncoder.register('cvae-endecoder')
class CVAEEnDeCoder(Seq2SeqEncoder):
    def __init__(self,
                 token_emb_dim: int,  # the dim of the token embedding
                 embedding_dropout: float = 0.0,
                 token_dropout: float = 0.0,  # dropout a token
                 token_proj_dim: int = None,  # the dim of projection of token embedding
                 use_x: bool = True,  # whether use x or not
                 combine_method: str = 'early_concat',
                 all_encoder: Seq2SeqEncoder = None,  # x, yin -> yout
                 x_encoder: Seq2SeqEncoder = None,  # x -> yout
                 yin_encoder: Seq2SeqEncoder = None  # yin -> yout
                 ) -> None:
        super(CVAEEnDeCoder, self).__init__()

        self.use_x = use_x
        # "early_concat" concat x emb and y emb and feed it to all_encoder
        # "late_concat" feed x emb to x_encoder and y emb to y_encoder and concat the output
        assert combine_method in {'early_concat', 'late_concat', 'only_x'}
        self.combine_method = combine_method

        # model
        # embedding
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self.token_dropout = Dropout2d(p=token_dropout)
        self.token_proj_dim = token_proj_dim
        if token_proj_dim:
            self.token_projection_layer = TimeDistributed(Linear(token_emb_dim, token_proj_dim))
        # encoder
        self.all_encoder = all_encoder
        self.x_encoder = x_encoder
        self.yin_encoder = yin_encoder

        # dimensionality
        # TODO: add dimensionality check?
        if self.use_x:
            if combine_method == 'early_concat':
                assert self.all_encoder, 'all_encoder not specified'
                self.input_dim = self.all_encoder.get_input_dim()
                self.output_dim = self.all_encoder.get_output_dim()
            elif combine_method == 'late_concat':
                assert self.x_encoder and self.yin_encoder, 'x_encoder or yin_encoder not specified'
                self.input_dim = self.x_encoder.get_input_dim() + self.yin_encoder.get_input_dim()
                self.output_dim = self.x_encoder.get_output_dim() + self.yin_encoder.get_output_dim()
            elif combine_method == 'only_x':
                assert self.x_encoder, 'x_encoder not specified'
                self.input_dim = self.x_encoder.get_input_dim()
                self.output_dim = self.x_encoder.get_output_dim()
        else:
            assert self.yin_encoder, 'yin_encoder not specified'
            self.input_dim = self.yin_encoder.get_input_dim()
            self.output_dim = self.yin_encoder.get_output_dim()


    def forward(self,
                t_emb: torch.Tensor,  # token emb
                v_emb: torch.Tensor,  # verb indicator emb
                yin_emb: torch.Tensor,  # input y
                mask: torch.LongTensor) -> torch.Tensor:
        # emb x
        if self.use_x:
            # token dropout
            t_emb = self.token_dropout(t_emb)  # TODO: is Dropout2d problematic?
            # token projection
            if self.token_proj_dim:
                t_emb = self.token_projection_layer(t_emb)
            if self.combine_method == 'early_concat':
                # SHAPE: (batch_size, seq_len, t_emb_dim + v_emb_dim + yin_emb_dim)
                inp_emb = torch.cat([t_emb, v_emb, yin_emb], -1)
            elif self.combine_method == 'late_concat' or self.combine_method == 'only_x':
                # SHAPE: (batch_size, seq_len, t_emb_dim + v_emb_dim)
                x_emb = torch.cat([t_emb, v_emb], -1)

        # encode
        if self.use_x:
            if self.combine_method == 'early_concat':
                inp_emb = self.embedding_dropout(inp_emb)
                enc = self.all_encoder(inp_emb, mask)
            elif self.combine_method == 'late_concat':
                x_emb = self.embedding_dropout(x_emb)
                yin_emb = self.embedding_dropout(yin_emb)
                enc = torch.cat([self.x_encoder(x_emb, mask), self.yin_encoder(yin_emb, mask)], -1)
            elif self.combine_method == 'only_x':
                enc = self.x_encoder(x_emb, mask)
        else:
            yin_emb = self.embedding_dropout(yin_emb)
            enc = self.yin_encoder(yin_emb, mask)

        return enc


    def get_input_dim(self) -> int:
        return self.input_dim


    def get_output_dim(self) -> int:
        return self.output_dim
