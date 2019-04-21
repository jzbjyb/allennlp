{
  "dataset_reader": {
    "type": "srl_mt",
    "file_format": "parallel",
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    },
    "lazy": true
  },
  "train_data_path": "data/openie/conll_for_allennlp/train_split_rm_coor/parallel/oie2016.train.all_tagging",
  "validation_data_path": "data/openie/conll_for_allennlp/dev_split_rm_coor/parallel/oie2016.dev.all_tagging",
  "model": {
    "type": "srl_oie_retag",
    "mode": "xsrl_oie",
    "text_field_embedder": {
      "elmo": {
        "type": "elmo_token_embedder",
        "options_file": "pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.options_file",
        "weight_file": "pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.weight_file",
        "do_layer_norm": false,
        "dropout": 0.1,
        "stateful": false,
        "scalar_requires_grad": false
      }
    },
    "encoder": {
      "type": "cvae-endecoder",
      "token_emb_dim": 1024,
      "embedding_dropout": 0.0,
      "token_dropout": 0.0,
      "token_proj_dim": null,
      "use_x": true,
      "combine_method": "mid_concat",
      "all_encoder_req_grad": true,
      "x_encoder_req_grad": false,
      "yin_encoder_req_grad": true,
      "x_encoder": {
        "type": "alternating_lstm",
        "input_size": 1124,
        "hidden_size": 300,
        "num_layers": 6,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      },
      "yin_encoder": {
        "type": "alternating_lstm",
        "input_size": 300,
        "hidden_size": 300,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      },
      "all_encoder": {
        "type": "alternating_lstm",
        "input_size": 600,
        "hidden_size": 300,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      }
    },
    "binary_feature_dim": 100,
    "binary_req_grad": false,
    "tag_feature_dim": 300,
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ],
      [
        "^(binary_feature_embedding\\..*|text_field_embedder\\..*)$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/multitask_vocab_shuffle/small_srl_oie_mt_large_task_encoder_sr3/best.th"
        }
      ],
      [
        "^encoder\\.x_encoder\\..*$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/multitask_vocab_shuffle/small_srl_oie_mt_large_task_encoder_sr3/best.th",
          "parameter_name_overrides": {
            "encoder.x_encoder._module.layer_0.input_linearity.weight": "encoder._module.layer_0.input_linearity.weight",
            "encoder.x_encoder._module.layer_0.state_linearity.bias": "encoder._module.layer_0.state_linearity.bias",
            "encoder.x_encoder._module.layer_0.state_linearity.weight": "encoder._module.layer_0.state_linearity.weight",
            "encoder.x_encoder._module.layer_1.input_linearity.weight": "encoder._module.layer_1.input_linearity.weight",
            "encoder.x_encoder._module.layer_1.state_linearity.bias": "encoder._module.layer_1.state_linearity.bias",
            "encoder.x_encoder._module.layer_1.state_linearity.weight": "encoder._module.layer_1.state_linearity.weight",
            "encoder.x_encoder._module.layer_2.input_linearity.weight": "encoder._module.layer_2.input_linearity.weight",
            "encoder.x_encoder._module.layer_2.state_linearity.bias": "encoder._module.layer_2.state_linearity.bias",
            "encoder.x_encoder._module.layer_2.state_linearity.weight": "encoder._module.layer_2.state_linearity.weight",
            "encoder.x_encoder._module.layer_3.input_linearity.weight": "encoder._module.layer_3.input_linearity.weight",
            "encoder.x_encoder._module.layer_3.state_linearity.bias": "encoder._module.layer_3.state_linearity.bias",
            "encoder.x_encoder._module.layer_3.state_linearity.weight": "encoder._module.layer_3.state_linearity.weight",
            "encoder.x_encoder._module.layer_4.input_linearity.weight": "encoder._module.layer_4.input_linearity.weight",
            "encoder.x_encoder._module.layer_4.state_linearity.bias": "encoder._module.layer_4.state_linearity.bias",
            "encoder.x_encoder._module.layer_4.state_linearity.weight": "encoder._module.layer_4.state_linearity.weight",
            "encoder.x_encoder._module.layer_5.input_linearity.weight": "encoder._module.layer_5.input_linearity.weight",
            "encoder.x_encoder._module.layer_5.state_linearity.bias": "encoder._module.layer_5.state_linearity.bias",
            "encoder.x_encoder._module.layer_5.state_linearity.weight": "encoder._module.layer_5.state_linearity.weight"
          }
        }
      ]
    ],
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "max_instances_in_memory": 800, // only shuffle consecutive 800 samples
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
  "vocabulary": { // use the multitask vocab
    "directory_path": "output/openie/vocab/srl_oie_multitask_large/"
  }
}
