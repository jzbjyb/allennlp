{
  "dataset_reader": {
    "type": "srl_mt",
    "default_task": "gt",
    "multiple_files": true, // use separate files for different tasks
    "restart_file": true, // iterate between tasks uniformly
    "task_weight": {"gt": 0.5, "neuoie": 0.5, "srl": 1.0},
    "multiple_files_sample_rate": [1, 1, 2],
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    },
    "lazy": true
  },
  "train_data_path": "data/openie/conll_for_allennlp/train_srl_oie_mt/oie/oie.shuffle.gold_conll:data/openie/conll_for_allennlp/neuoie/neuoie/neuoie_10000.shuffle.gold_conll:data/openie/conll_for_allennlp/train_srl_oie_mt/srl/ontonotes.shuffle.gold_conll",
  "validation_data_path": "data/openie/conll_for_allennlp/dev_srl_oie_mt/oie/oie.shuffle.gold_conll",
  "test_data_path": "data/openie/conll_for_allennlp/test_split_rm_coor/oie2016.test.gold_conll",
  "model": {
    "type": "srl_mt",
    "text_field_embedder": {
      "elmo": {
        "type": "elmo_token_embedder",
        "options_file": "/home/zhengbaj/exp/allennlp/pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.options_file",
        "weight_file": "/home/zhengbaj/exp/allennlp/pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.weight_file",
        "do_layer_norm": false,
        "dropout": 0.1,
        "stateful": false
      }
    },
    "initializer": [
      [
        "^(gt_tag_projection_layer.*weight|neuoie_tag_projection_layer.*weight)$", // openie tag proj layer
        {
          "type": "orthogonal"
        }
      ],
      [
        "srl_tag_projection_layer.*", // srl tag proj layer
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
        "^(encoder.*|binary_feature_embedding.*|text_field_embedder.*)$", // shared encoder
        {
          "type": "pretrained",
          "weights_file_path": "/home/zhengbaj/exp/allennlp/output/srl_official_srl_small/best.th"
        }
      ],
      [
        "^(gt_task_encoder.*|srl_task_encoder.*|neuoie_task_encoder.*)$", // task encoder
        {
          "type": "pretrained",
          "weights_file_path": "/home/zhengbaj/exp/allennlp/output/srl_official_srl_small/best.th",
          "parameter_name_overrides": {
            "gt_task_encoder._module.layer_0.input_linearity.weight": "encoder._module.layer_2.input_linearity.weight",
            "gt_task_encoder._module.layer_0.state_linearity.weight": "encoder._module.layer_2.state_linearity.weight",
            "gt_task_encoder._module.layer_0.state_linearity.bias": "encoder._module.layer_2.state_linearity.bias",
            "gt_task_encoder._module.layer_1.input_linearity.weight": "encoder._module.layer_3.input_linearity.weight",
            "gt_task_encoder._module.layer_1.state_linearity.weight": "encoder._module.layer_3.state_linearity.weight",
            "gt_task_encoder._module.layer_1.state_linearity.bias": "encoder._module.layer_3.state_linearity.bias",
            "neuoie_task_encoder._module.layer_0.input_linearity.weight": "encoder._module.layer_2.input_linearity.weight",
            "neuoie_task_encoder._module.layer_0.state_linearity.weight": "encoder._module.layer_2.state_linearity.weight",
            "neuoie_task_encoder._module.layer_0.state_linearity.bias": "encoder._module.layer_2.state_linearity.bias",
            "neuoie_task_encoder._module.layer_1.input_linearity.weight": "encoder._module.layer_3.input_linearity.weight",
            "neuoie_task_encoder._module.layer_1.state_linearity.weight": "encoder._module.layer_3.state_linearity.weight",
            "neuoie_task_encoder._module.layer_1.state_linearity.bias": "encoder._module.layer_3.state_linearity.bias",
            "srl_task_encoder._module.layer_0.input_linearity.weight": "encoder._module.layer_2.input_linearity.weight",
            "srl_task_encoder._module.layer_0.state_linearity.weight": "encoder._module.layer_2.state_linearity.weight",
            "srl_task_encoder._module.layer_0.state_linearity.bias": "encoder._module.layer_2.state_linearity.bias",
            "srl_task_encoder._module.layer_1.input_linearity.weight": "encoder._module.layer_3.input_linearity.weight",
            "srl_task_encoder._module.layer_1.state_linearity.weight": "encoder._module.layer_3.state_linearity.weight",
            "srl_task_encoder._module.layer_1.state_linearity.bias": "encoder._module.layer_3.state_linearity.bias"
          }
        }
      ]
    ],
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 1124,
      "hidden_size": 64,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.1,
      "use_input_projection_bias": false
    },
    "task_encoder": {
      "gt": {
        "type": "alternating_lstm",
        "input_size": 64,
        "hidden_size": 64,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      },
      "neuoie": {
        "type": "alternating_lstm",
        "input_size": 64,
        "hidden_size": 64,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      },
      "srl": {
        "type": "alternating_lstm",
        "input_size": 64,
        "hidden_size": 64,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      }
    },
    "encoder_requires_grad": true,
    "task_encoder_requires_grad": {
      "gt": true,
      "neuoie": true,
      "srl": true
    },
    "binary_feature_dim": 100,
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "task_bucket",
    "max_instances_in_memory": 6080, // only shuffle consecutive 800 samples
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
    "directory_path": "output/openie/vocab/srl_oie_neuoie_multitask_small/"
  },
  "evaluate_on_test": true
}
