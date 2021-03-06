// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader": {
    "type": "srl_mt",
    "default_task": "gt",
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    }
  },
  //"train_data_path": "/home/zhengbaj/exp/allennlp/data/srl/conll-formatted-ontonotes-5.0/data/train/",
  "train_data_path": "/home/zhengbaj/exp/allennlp/data/openie/conll_for_allennlp/train_mt_only_openie4",
  //"validation_data_path": "/home/zhengbaj/exp/allennlp/data/srl/conll-formatted-ontonotes-5.0/data/development/",
  "validation_data_path": "/home/zhengbaj/exp/allennlp/data/openie/conll_for_allennlp/dev_mt_only_openie4",
  "model": {
    "type": "srl_mt",
    "text_field_embedder": {
      "elmo": {
        "type": "elmo_token_embedder",
        "options_file": "/home/zhengbaj/exp/allennlp/pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.options_file",
        "weight_file": "/home/zhengbaj/exp/allennlp/pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.weight_file",
        "do_layer_norm": false,
        "dropout": 0.1
      }
    },
    "initializer": [
      [
        "tag_projection_layer_mt.*weight",
        {
          "type": "orthogonal"
        }
      ],
      [
        "^((?!(tag_projection_layer_mt|task_encoder)).)*$",
        {
          "type": "pretrained",
          "weights_file_path": "/home/zhengbaj/exp/allennlp/output/srl_official_srl_small/best.th"
        }
      ]
    ],
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 1124,
      "hidden_size": 64,
      "num_layers": 4,
      "recurrent_dropout_probability": 0.1,
      "use_input_projection_bias": false
    },
    "task_encoder": {
      "type": "alternating_lstm",
      "input_size": 64,
      "hidden_size": 64,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.1,
      "use_input_projection_bias": false
    },
    "encoder_requires_grad": false,
    "task_encoder_requires_grad": true,
    "binary_feature_dim": 100,
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },
  "trainer": {
    "num_epochs": 200,
    "grad_clipping": 1.0,
    "patience": 10,
    "num_serialized_models_to_keep": 10,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 0,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  },
  //"vocabulary": {
  //  "directory_path": "/home/zhengbaj/exp/allennlp/pretrain/srl-model-2018.05.25/vocabulary_for_openie_finetune/"
  //}
}
