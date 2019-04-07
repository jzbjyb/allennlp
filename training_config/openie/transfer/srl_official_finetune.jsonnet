{
  "dataset_reader": {
    "type": "srl",
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    }
  },
  "train_data_path": "data/openie/conll_for_allennlp/train_split_rm_coor",
  "validation_data_path": "data/openie/conll_for_allennlp/dev_split_rm_coor",
  "model": {
    "type": "srl",
    "text_field_embedder": {
      "elmo": {
        "type": "elmo_token_embedder",
        "options_file": "pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.options_file",
        "weight_file": "pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.weight_file",
        "do_layer_norm": false,
        "dropout": 0.1
      }
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ],
      [
        "^((?!tag_projection_layer).)*$",
        {
          "type": "pretrained",
          "weights_file_path": "pretrain/srl-model-2018.05.25/weights.th"
        }
       ]
    ],
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 1124,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_input_projection_bias": false
    },
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
  }
}
