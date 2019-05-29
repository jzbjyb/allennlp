// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader": {
    "type": "srl",
    "domain_identifier": "nw"
  },
  "train_data_path": "/home/zhengbaj/exp/allennlp/data/srl/conll-formatted-ontonotes-5.0/data/train/",
  "validation_data_path": "/home/zhengbaj/exp/allennlp/data/srl/conll-formatted-ontonotes-5.0/data/development/",
  "test_data_path": "/home/zhengbaj/exp/allennlp/data/srl/conll-formatted-ontonotes-5.0/data/test/",
  "model": {
    "type": "srl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
            "trainable": true
        }
      }
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 200,
      "hidden_size": 256,
      "num_layers": 4,
      "recurrent_dropout_probability": 0.05,
      "use_highway": true
    },
    "binary_feature_dim": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },
  "trainer": {
    "num_epochs": 50,
    "grad_clipping": 1.0,
    "patience": 5,
    "num_serialized_models_to_keep": 10,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 0,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
