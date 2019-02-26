// Configuration for a rereanking model from small SRL-openie model (no multitask)
{
  "dataset_reader": {
    "type": "rerank",
    "default_task": "gt",
    "one_verb": true,
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    }
  },
  "train_data_path": "/home/zhengbaj/exp/allennlp/data/openie/rerank/small_finetune/oie2016.train.beam.conll",
  "validation_data_path": "/home/zhengbaj/exp/allennlp/data/openie/rerank/small_finetune/oie2016.dev.beam.conll",
  "model": {
    "type": "reranker",
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
        "^((?!score_layer).)*$",
        {
          "type": "pretrained",
          "weights_file_path": "/home/zhengbaj/exp/allennlp/output/srl_official_small_finetune_split_rm_coor/best.th",
          "parameter_name_overrides": {
            "tag_projection_layer_mt._module.weight": "tag_projection_layer._module.weight",
            "tag_projection_layer_mt._module.bias": "tag_projection_layer._module.bias"
          }
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
    "encoder_requires_grad": true,
    "binary_feature_dim": 100,
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },
  "trainer": {
    "num_epochs": 100,
    "grad_clipping": 1.0,
    "patience": 5,
    "num_serialized_models_to_keep": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  },
  "vocabulary": {
    "directory_path": "/home/zhengbaj/exp/allennlp/pretrain/srl-model-2018.05.25/vocabulary_for_openie_finetune/",
    "extend": true
  }
}
