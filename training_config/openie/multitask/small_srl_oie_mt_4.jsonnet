{
  "dataset_reader": {
    "type": "srl_mt",
    "default_task": "gt",
    "multiple_files": true, // use separate files for different tasks
    "restart_file": true, // iterate between tasks uniformly
    "task_weight": {"gt": 1.6, "srl": 0.4},
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    },
    "lazy": true
  },
  "train_data_path": "data/openie/conll_for_allennlp/train_srl_oie_mt/oie/oie.gold_conll:data/openie/conll_for_allennlp/train_srl_oie_mt/srl/ontonotes.gold_conll",
  "validation_data_path": "data/openie/conll_for_allennlp/dev_srl_oie_mt/oie/oie.gold_conll",
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
        "gt_tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ],
      [
        "srl_tag_projection_layer.*",
        {
          "type": "pretrained",
          "weights_file_path": "/home/zhengbaj/exp/allennlp/output/srl_official_srl_small/best.th",
          "parameter_name_overrides": {
            "srl_tag_projection_layer._module.weight": "tag_projection_layer._module.weight",
            "srl_tag_projection_layer._module.bias": "tag_projection_layer._module.bias"
          }
        }
      ],
      [
        "^((?!(tag_projection_layer|task_encoder)).)*$",
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
    "encoder_requires_grad": true,
    "task_encoder_requires_grad": false,
    "binary_feature_dim": 100,
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "bucket",
    "max_instances_in_memory": 6080, // only shuffle consecutive 6080 samples
    "instances_per_epoch": 6080, // we only have 3k oie training samples
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },
  "validation_iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },
  "trainer": {
    "num_epochs": 200,
    "grad_clipping": 1.0,
    "patience": 10,
    "num_serialized_models_to_keep": 10,
    "validation_metric": "+gt_f1-measure-overall",
    "cuda_device": 0,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  },
  "vocabulary": {
    "directory_path": "output/openie/vocab/srl_oie_multitask_small/"
  }
}
