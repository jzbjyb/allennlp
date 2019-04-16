from typing import Dict
from overrides import overrides

import torch

from allennlp.models.model import Model
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode


class BaseModel(Model):
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
        if self.use_crf:
            # TODO: add prob
            output_dict['tags'] = [
                [self.vocab.get_token_from_index(tag, namespace='labels') for tag in instance_tags]
                for instance_tags in output_dict["tags"]]
            return output_dict
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags_ind, all_tags, all_probs = [], [], []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            lp = torch.log(predictions[:length])  # log prob is required by viterbi decoding
            max_likelihood_sequence, score = viterbi_decode(lp, transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace='MT_gt_labels')
                    for x in max_likelihood_sequence]  # TODO: add more task and avoid "gt"
            probs = [predictions[i, max_likelihood_sequence[i]].numpy().tolist()
                     for i in range(len(max_likelihood_sequence))]
            all_tags_ind.append(max_likelihood_sequence)
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
        # TODO: add more task and avoid "gt"
        all_labels = self.vocab.get_index_to_token_vocabulary('MT_gt_labels')
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float('-inf')
        return transition_matrix