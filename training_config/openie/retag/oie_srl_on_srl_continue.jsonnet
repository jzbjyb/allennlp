{
  "dataset_reader": {
    "type": "srl_mt",
    "file_format": "parallel",
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    },
    "lazy": true
  },
  "train_data_path": "data/openie/conll_for_allennlp/train_parallel_on_srl3/oie2016.train.shuffle.all_tagging",
  "validation_data_path": "data/openie/conll_for_allennlp/dev_parallel_on_srl3/oie2016.dev.shuffle.all_tagging",
  "model": {
    "type": "srl_oie_retag",
    "mode": "oie_srl",
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
      "use_x": false,
      "yin_encoder": {
        "type": "alternating_lstm",
        "input_size": 64,
        "hidden_size": 64,
        "num_layers": 4,
        "recurrent_dropout_probability": 0.0,
        "use_input_projection_bias": false
      }
    },
    "binary_feature_dim": 100,
    "binary_req_grad": false,
    "tag_feature_dim": 64,
    "tag_proj_req_grad": false,
    "initializer": [
      [
        "^.*$",
        {
          "type": "pretrained",
          "weights_file_path": "output/openie/retag/oie_srl_middle/best.th"
        }
      ]
    ],
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "max_instances_in_memory": 5000, // only shuffle consecutive 800 samples
    "instances_per_epoch": 20000, // 250k train
    "batch_size" : 256
  },
  "validation_iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "max_instances_in_memory": 5000, // only shuffle consecutive 800 samples
    "instances_per_epoch": 10000, // 35k dev
    "batch_size" : 256
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
    "directory_path": "output/openie/vocab/srl_oie_multitask_middle/"
  }
}
