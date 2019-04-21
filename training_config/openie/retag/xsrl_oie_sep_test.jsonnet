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
        "stateful": false
      }
    },
    "encoder": {
      "type": "cvae-endecoder",
      "token_emb_dim": 1024,
      "embedding_dropout": 0.0,
      "token_dropout": 0.0,
      "token_proj_dim": null,
      "use_x": true,
      "combine_method": "late_concat",
      "x_encoder": {
        "type": "alternating_lstm",
        "input_size": 1124,
        "hidden_size": 300,
        "num_layers": 8,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      },
      "yin_encoder": {
        "type": "alternating_lstm",
        "input_size": 64,
        "hidden_size": 64,
        "num_layers": 4,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      }
    },
    "binary_feature_dim": 100,
    "tag_feature_dim": 64,
    "initializer": [
      [  // input layer
        "^(binary_feature_embedding\\..*|text_field_embedder\\..*|tag_feature_embedding\\..*)$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/semi_vae/large_sep_model_rerun/best.th",
          "parameter_name_overrides": {
            "binary_feature_embedding.weight": "enc_bin_emb.weight",
            "text_field_embedder.token_embedder_elmo._elmo.scalar_mix_0.gamma": "enc_post_elmo_.scalar_mix_0.gamma",
            "text_field_embedder.token_embedder_elmo._elmo.scalar_mix_0.scalar_parameters.0": "enc_post_elmo_.scalar_mix_0.scalar_parameters.0",
            "text_field_embedder.token_embedder_elmo._elmo.scalar_mix_0.scalar_parameters.1": "enc_post_elmo_.scalar_mix_0.scalar_parameters.1",
            "text_field_embedder.token_embedder_elmo._elmo.scalar_mix_0.scalar_parameters.2": "enc_post_elmo_.scalar_mix_0.scalar_parameters.2",
            "tag_feature_embedding.weight": "y2_embedding.weight"
          }
        }
      ],
      [  // output layer
        "^tag_projection_layer\\..*$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/semi_vae/large_sep_model_rerun/best.th",
          "parameter_name_overrides": {
            "tag_projection_layer._module.weight": "enc_y1_proj._module.weight",
            "tag_projection_layer._module.bias": "enc_y1_proj._module.bias"
          }
        }
      ],
      [  // model layer
        "^(encoder\\.x_encoder\\..*|encoder\\.yin_encoder\\..*)$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/semi_vae/large_sep_model_rerun/best.th"
        }
      ]
    ],
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "instances_per_epoch": 0,
    "batch_size" : 80
  },
  "validation_iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },
  "trainer": {
    "num_epochs": 1,
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
