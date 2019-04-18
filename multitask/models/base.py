from typing import Dict
from overrides import overrides

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode


class BaseModel(Model):
    def __init__(self,
                 decode_namespace: str,
                 vocab: Vocabulary,
                 regularizer: RegularizerApplicator = None) -> None:
        super(BaseModel, self).__init__(vocab, regularizer)
        self._decode_namespace = decode_namespace


    def get_decode_pseudo_class_prob(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Use decode to get tag results used in test-time and get their pseudo class probability.
        '''
        output_dict = self.decode(output_dict)
        cp = torch.zeros_like(output_dict['class_probabilities'])
        for i, tags in enumerate(output_dict['tags_ind']):
            for j, t in enumerate(tags):
                cp[i, j, t] = 1.0
        return cp


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        # crf
        if self.use_crf:
            # TODO: add prob
            output_dict['tags'] = [
                [self.vocab.get_token_from_index(tag, namespace='labels') for tag in instance_tags]
                for instance_tags in output_dict['tags']]
            return output_dict

        # independent prediction
        cp = output_dict['class_probabilities']
        seq_len = get_lengths_from_binary_sequence_mask(output_dict['mask']).data.tolist()
        if cp.dim() == 3:
            cp_list = [cp[i].detach().cpu() for i in range(cp.size(0))]
        else:
            cp_list = [cp]
        all_tags_ind, all_tags, all_probs = [], [], []
        trans_mat = self.get_viterbi_pairwise_potentials()
        for cp, length in zip(cp_list, seq_len):
            lp = torch.log(cp[:length] + 1e-10)  # log prob is required by viterbi decoding
            best_seq, score = viterbi_decode(lp, trans_mat)
            tags = [self.vocab.get_token_from_index(x, namespace=self._decode_namespace) for x in best_seq]
            probs = [cp[i, best_seq[i]].numpy().tolist() for i in range(len(best_seq))]
            all_tags_ind.append(best_seq)
            all_tags.append(tags)
            all_probs.append(probs)
        output_dict['tags_ind'] = all_tags_ind
        output_dict['tags'] = all_tags
        output_dict['probs'] = all_probs
        return output_dict


    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary(self._decode_namespace)
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float('-inf')
        return transition_matrix