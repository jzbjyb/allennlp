{
  "dataset_reader": {
    "type": "srl_mt",
    "default_task": "gt",
    "multiple_files": true, // use separate files for different tasks
    "restart_file": true, // iterate between tasks uniformly
    "task_weight": {"gt": 1.0, "srl": 1.0},
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    },
    "lazy": true
  },
  "train_data_path": "data/openie/conll_for_allennlp/train_srl_oie_mt/oie/oie.shuffle.gold_conll:data/openie/conll_for_allennlp/train_srl_oie_mt/srl_nw_wsj/ontonotes.shuffle.gold_conll",
  "validation_data_path": "data/openie/conll_for_allennlp/dev_srl_oie_mt/oie/oie.shuffle.gold_conll",
  "test_data_path": "data/openie/conll_for_allennlp/test_split_rm_coor/oie2016.test.gold_conll",
  "model": {
    "type": "semi_cvae_oie",
    "y1_ns": "gt",
    "y2_ns": "srl",
    "kl_method": "exact",
    "sample_num": 5,
    "sample_algo": "beam",
    "infer_algo": "reinforce",
    "use_rule": "o_boundary_rule",
    "baseline": "wb",
    "clip_reward": 10.0,
    "temperature": 1.0,
    "beta": 1.0,
    "unsup_loss_type": "all",
    "decode_span_metric": true,
    "share_param": true,
    "encoder_same_as_pior": true,
    "dec_skip": [
      "_module.layer_6.input_linearity.weight",
      "_module.layer_6.state_linearity.weight",
      "_module.layer_6.state_linearity.bias",
      "_module.layer_7.input_linearity.weight",
      "_module.layer_7.state_linearity.weight",
      "_module.layer_7.state_linearity.bias"],
    "text_field_embedder": {
      "elmo": {
        "type": "elmo_token_embedder",
        "options_file": "pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.options_file",
        "weight_file": "pretrain/srl-model-2018.05.25/fta/model.text_field_embedder.elmo.weight_file",
        "do_layer_norm": false,
        "dropout": 0.1,
        "stateful": false,
        "use_post_elmo": false,
        "scalar_requires_grad": false
      }
    },
    "discriminator": {
      "type": "alternating_lstm",
      "input_size": 1124,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_input_projection_bias": false
    },
    "encoder": {
      "type": "cvae-endecoder",
      "token_emb_dim": 1024,
      "embedding_dropout": 0.0,
      "token_dropout": 0.0,
      "token_proj_dim": null,
      "use_x": true,
      "combine_method": "only_x",
      "x_encoder": {
        "type": "alternating_lstm",
        "input_size": 1124,
        "hidden_size": 300,
        "num_layers": 8,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      }
    },
    "decoder": {
      "type": "cvae-endecoder",
      "token_emb_dim": 1024,
      "embedding_dropout": 0.0,
      "token_dropout": 0.0,
      "token_proj_dim": null,
      "use_x": true,
      "combine_method": "late_add",
      "late_add_alpha": 0.0,
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
        "input_size": 300,
        "hidden_size": 300,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.1,
        "use_input_projection_bias": false
      }
    },
    "binary_feature_dim": 100,
    "y_feature_dim": 300,
    "initializer": [
      [ // token emb and verb emb
        "^(text_field_embedder\\..*|binary_feature_embedding\\..*)$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/multitask_vocab_shuffle/small_srl_oie_mt_large_task_encoder_sr3/best.th"
        }
      ],
      [ // use pretrained model from multitask learning to init discriminator
        "^(discriminator\\..*|disc_y1_proj\\..*)$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/multitask_vocab_shuffle/small_srl_oie_mt_large_task_encoder_sr3/best.th",
          "parameter_name_overrides": {
            "discriminator._module.layer_0.input_linearity.weight": "encoder._module.layer_0.input_linearity.weight",
            "discriminator._module.layer_0.state_linearity.weight": "encoder._module.layer_0.state_linearity.weight",
            "discriminator._module.layer_0.state_linearity.bias": "encoder._module.layer_0.state_linearity.bias",
            "discriminator._module.layer_1.input_linearity.weight": "encoder._module.layer_1.input_linearity.weight",
            "discriminator._module.layer_1.state_linearity.weight": "encoder._module.layer_1.state_linearity.weight",
            "discriminator._module.layer_1.state_linearity.bias": "encoder._module.layer_1.state_linearity.bias",
            "discriminator._module.layer_2.input_linearity.weight": "encoder._module.layer_2.input_linearity.weight",
            "discriminator._module.layer_2.state_linearity.weight": "encoder._module.layer_2.state_linearity.weight",
            "discriminator._module.layer_2.state_linearity.bias": "encoder._module.layer_2.state_linearity.bias",
            "discriminator._module.layer_3.input_linearity.weight": "encoder._module.layer_3.input_linearity.weight",
            "discriminator._module.layer_3.state_linearity.weight": "encoder._module.layer_3.state_linearity.weight",
            "discriminator._module.layer_3.state_linearity.bias": "encoder._module.layer_3.state_linearity.bias",
            "discriminator._module.layer_4.input_linearity.weight": "encoder._module.layer_4.input_linearity.weight",
            "discriminator._module.layer_4.state_linearity.weight": "encoder._module.layer_4.state_linearity.weight",
            "discriminator._module.layer_4.state_linearity.bias": "encoder._module.layer_4.state_linearity.bias",
            "discriminator._module.layer_5.input_linearity.weight": "encoder._module.layer_5.input_linearity.weight",
            "discriminator._module.layer_5.state_linearity.weight": "encoder._module.layer_5.state_linearity.weight",
            "discriminator._module.layer_5.state_linearity.bias": "encoder._module.layer_5.state_linearity.bias",
            "discriminator._module.layer_6.input_linearity.weight": "gt_task_encoder._module.layer_0.input_linearity.weight",
            "discriminator._module.layer_6.state_linearity.weight": "gt_task_encoder._module.layer_0.state_linearity.weight",
            "discriminator._module.layer_6.state_linearity.bias": "gt_task_encoder._module.layer_0.state_linearity.bias",
            "discriminator._module.layer_7.input_linearity.weight": "gt_task_encoder._module.layer_1.input_linearity.weight",
            "discriminator._module.layer_7.state_linearity.weight": "gt_task_encoder._module.layer_1.state_linearity.weight",
            "discriminator._module.layer_7.state_linearity.bias": "gt_task_encoder._module.layer_1.state_linearity.bias",
            "disc_y1_proj._module.weight": "gt_tag_projection_layer._module.weight",
            "disc_y1_proj._module.bias": "gt_tag_projection_layer._module.bias"
          }
        }
      ],
      [ // use pretrained model from retag model to init decoder
        "^(decoder\\.x_encoder\\..*|dec_y2_proj\\..*)$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/multitask_vocab_shuffle/small_srl_oie_mt_large_task_encoder_sr3/best.th",
          "parameter_name_overrides": {
            "decoder.x_encoder._module.layer_0.input_linearity.weight": "encoder._module.layer_0.input_linearity.weight",
            "decoder.x_encoder._module.layer_0.state_linearity.weight": "encoder._module.layer_0.state_linearity.weight",
            "decoder.x_encoder._module.layer_0.state_linearity.bias": "encoder._module.layer_0.state_linearity.bias",
            "decoder.x_encoder._module.layer_1.input_linearity.weight": "encoder._module.layer_1.input_linearity.weight",
            "decoder.x_encoder._module.layer_1.state_linearity.weight": "encoder._module.layer_1.state_linearity.weight",
            "decoder.x_encoder._module.layer_1.state_linearity.bias": "encoder._module.layer_1.state_linearity.bias",
            "decoder.x_encoder._module.layer_2.input_linearity.weight": "encoder._module.layer_2.input_linearity.weight",
            "decoder.x_encoder._module.layer_2.state_linearity.weight": "encoder._module.layer_2.state_linearity.weight",
            "decoder.x_encoder._module.layer_2.state_linearity.bias": "encoder._module.layer_2.state_linearity.bias",
            "decoder.x_encoder._module.layer_3.input_linearity.weight": "encoder._module.layer_3.input_linearity.weight",
            "decoder.x_encoder._module.layer_3.state_linearity.weight": "encoder._module.layer_3.state_linearity.weight",
            "decoder.x_encoder._module.layer_3.state_linearity.bias": "encoder._module.layer_3.state_linearity.bias",
            "decoder.x_encoder._module.layer_4.input_linearity.weight": "encoder._module.layer_4.input_linearity.weight",
            "decoder.x_encoder._module.layer_4.state_linearity.weight": "encoder._module.layer_4.state_linearity.weight",
            "decoder.x_encoder._module.layer_4.state_linearity.bias": "encoder._module.layer_4.state_linearity.bias",
            "decoder.x_encoder._module.layer_5.input_linearity.weight": "encoder._module.layer_5.input_linearity.weight",
            "decoder.x_encoder._module.layer_5.state_linearity.weight": "encoder._module.layer_5.state_linearity.weight",
            "decoder.x_encoder._module.layer_5.state_linearity.bias": "encoder._module.layer_5.state_linearity.bias",
            "decoder.x_encoder._module.layer_6.input_linearity.weight": "srl_task_encoder._module.layer_0.input_linearity.weight",
            "decoder.x_encoder._module.layer_6.state_linearity.weight": "srl_task_encoder._module.layer_0.state_linearity.weight",
            "decoder.x_encoder._module.layer_6.state_linearity.bias": "srl_task_encoder._module.layer_0.state_linearity.bias",
            "decoder.x_encoder._module.layer_7.input_linearity.weight": "srl_task_encoder._module.layer_1.input_linearity.weight",
            "decoder.x_encoder._module.layer_7.state_linearity.weight": "srl_task_encoder._module.layer_1.state_linearity.weight",
            "decoder.x_encoder._module.layer_7.state_linearity.bias": "srl_task_encoder._module.layer_1.state_linearity.bias",
            "dec_y2_proj._module.weight": "srl_tag_projection_layer._module.weight",
            "dec_y2_proj._module.bias": "srl_tag_projection_layer._module.bias"
          }
        }
      ],
      [
        "^(decoder\\.yin_encoder\\..*|y1_embedding\\..*)$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/retag/oie_srl_on_real_on_srl_continue/best.th",
          "parameter_name_overrides": {
            "y1_embedding.weight": "tag_feature_embedding.weight"
          }
        }
      ],
      [ // use pretrained model from retag model to init encoder
        "^(encoder\\..*|enc_y1_proj\\..*)$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/multitask_vocab_shuffle/small_srl_oie_mt_large_task_encoder_sr3/best.th",
          "parameter_name_overrides": {
            "encoder.x_encoder._module.layer_0.input_linearity.weight": "encoder._module.layer_0.input_linearity.weight",
            "encoder.x_encoder._module.layer_0.state_linearity.weight": "encoder._module.layer_0.state_linearity.weight",
            "encoder.x_encoder._module.layer_0.state_linearity.bias": "encoder._module.layer_0.state_linearity.bias",
            "encoder.x_encoder._module.layer_1.input_linearity.weight": "encoder._module.layer_1.input_linearity.weight",
            "encoder.x_encoder._module.layer_1.state_linearity.weight": "encoder._module.layer_1.state_linearity.weight",
            "encoder.x_encoder._module.layer_1.state_linearity.bias": "encoder._module.layer_1.state_linearity.bias",
            "encoder.x_encoder._module.layer_2.input_linearity.weight": "encoder._module.layer_2.input_linearity.weight",
            "encoder.x_encoder._module.layer_2.state_linearity.weight": "encoder._module.layer_2.state_linearity.weight",
            "encoder.x_encoder._module.layer_2.state_linearity.bias": "encoder._module.layer_2.state_linearity.bias",
            "encoder.x_encoder._module.layer_3.input_linearity.weight": "encoder._module.layer_3.input_linearity.weight",
            "encoder.x_encoder._module.layer_3.state_linearity.weight": "encoder._module.layer_3.state_linearity.weight",
            "encoder.x_encoder._module.layer_3.state_linearity.bias": "encoder._module.layer_3.state_linearity.bias",
            "encoder.x_encoder._module.layer_4.input_linearity.weight": "encoder._module.layer_4.input_linearity.weight",
            "encoder.x_encoder._module.layer_4.state_linearity.weight": "encoder._module.layer_4.state_linearity.weight",
            "encoder.x_encoder._module.layer_4.state_linearity.bias": "encoder._module.layer_4.state_linearity.bias",
            "encoder.x_encoder._module.layer_5.input_linearity.weight": "encoder._module.layer_5.input_linearity.weight",
            "encoder.x_encoder._module.layer_5.state_linearity.weight": "encoder._module.layer_5.state_linearity.weight",
            "encoder.x_encoder._module.layer_5.state_linearity.bias": "encoder._module.layer_5.state_linearity.bias",
            "encoder.x_encoder._module.layer_6.input_linearity.weight": "gt_task_encoder._module.layer_0.input_linearity.weight",
            "encoder.x_encoder._module.layer_6.state_linearity.weight": "gt_task_encoder._module.layer_0.state_linearity.weight",
            "encoder.x_encoder._module.layer_6.state_linearity.bias": "gt_task_encoder._module.layer_0.state_linearity.bias",
            "encoder.x_encoder._module.layer_7.input_linearity.weight": "gt_task_encoder._module.layer_1.input_linearity.weight",
            "encoder.x_encoder._module.layer_7.state_linearity.weight": "gt_task_encoder._module.layer_1.state_linearity.weight",
            "encoder.x_encoder._module.layer_7.state_linearity.bias": "gt_task_encoder._module.layer_1.state_linearity.bias",
            "enc_y1_proj._module.weight": "gt_tag_projection_layer._module.weight",
            "enc_y1_proj._module.bias": "gt_tag_projection_layer._module.bias"
          }
        }
      ]
    ],
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "task_bucket",
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
    "validation_metric": "+y1_f1-measure-overall",
    "cuda_device": 0,
    //"optimizer": {
    //  "type": "adadelta",
    //  "rho": 0.95,
    //  "lr": 0.1
    //},
    "optimizer": {
      "type": "sgd",
      "lr": 0.0001
    }
  },
  "vocabulary": {
    "directory_path": "output/openie/vocab/srl_oie_multitask_large/"
  },
  "evaluate_on_test": true
}
