---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:19604
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: python check if key value is not in array of values
  sentences:
  - "def isin(value, values):\n    \"\"\" Check that value is in values \"\"\"\n \
    \   for i, v in enumerate(value):\n        if v not in np.array(values)[:, i]:\n\
    \            return False\n    return True"
  - "def _linearInterpolationTransformMatrix(matrix1, matrix2, value):\n    \"\"\"\
    \ Linear, 'oldstyle' interpolation of the transform matrix.\"\"\"\n    return\
    \ tuple(_interpolateValue(matrix1[i], matrix2[i], value) for i in range(len(matrix1)))"
  - "def index_nearest(value, array):\n    \"\"\"\n    expects a _n.array\n    returns\
    \ the global minimum of (value-array)^2\n    \"\"\"\n\n    a = (array-value)**2\n\
    \    return index(a.min(), a)"
- source_sentence: python make url query string
  sentences:
  - "def create_search_url(self):\n        \"\"\" Generates (urlencoded) query string\
    \ from stored key-values tuples\n\n        :returns: A string containing all arguments\
    \ in a url-encoded format\n        \"\"\"\n\n        url = '?'\n        for key,\
    \ value in self.arguments.items():\n            url += '%s=%s&' % (quote_plus(key),\
    \ quote_plus(value))\n        self.url = url[:-1]\n        return self.url"
  - "def launched():\n    \"\"\"Test whether the current python environment is the\
    \ correct lore env.\n\n    :return:  :any:`True` if the environment is launched\n\
    \    :rtype: bool\n    \"\"\"\n    if not PREFIX:\n        return False\n\n  \
    \  return os.path.realpath(sys.prefix) == os.path.realpath(PREFIX)"
  - "def exists(self):\n        \"\"\"\n        Performs an existence check on the\
    \ remote database.\n\n        :returns: Boolean True if the database exists, False\
    \ otherwise\n        \"\"\"\n        resp = self.r_session.head(self.database_url)\n\
    \        if resp.status_code not in [200, 404]:\n            resp.raise_for_status()\n\
    \n        return resp.status_code == 200"
- source_sentence: python normalize text string
  sentences:
  - "def split(text: str) -> List[str]:\n    \"\"\"Split a text into a list of tokens.\n\
    \n    :param text: the text to split\n    :return: tokens\n    \"\"\"\n    return\
    \ [word for word in SEPARATOR.split(text) if word.strip(' \\t')]"
  - "def norm(x, mu, sigma=1.0):\n    \"\"\" Scipy norm function \"\"\"\n    return\
    \ stats.norm(loc=mu, scale=sigma).pdf(x)"
  - "def normalize(self, string):\n        \"\"\"Normalize the string according to\
    \ normalization list\"\"\"\n        return ''.join([self._normalize.get(x, x)\
    \ for x in nfd(string)])"
- source_sentence: how to visualize decision tree python sklearn
  sentences:
  - "def human__decision_tree():\n    \"\"\" Decision Tree\n    \"\"\"\n\n    # build\
    \ data\n    N = 1000000\n    M = 3\n    X = np.zeros((N,M))\n    X.shape\n   \
    \ y = np.zeros(N)\n    X[0, 0] = 1\n    y[0] = 8\n    X[1, 1] = 1\n    y[1] =\
    \ 8\n    X[2, 0:2] = 1\n    y[2] = 4\n\n    # fit model\n    xor_model = sklearn.tree.DecisionTreeRegressor(max_depth=2)\n\
    \    xor_model.fit(X, y)\n\n    return xor_model"
  - "def vectorize(values):\n    \"\"\"\n    Takes a value or list of values and returns\
    \ a single result, joined by \",\"\n    if necessary.\n    \"\"\"\n    if isinstance(values,\
    \ list):\n        return ','.join(str(v) for v in values)\n    return values"
  - "def is_quoted(arg: str) -> bool:\n    \"\"\"\n    Checks if a string is quoted\n\
    \    :param arg: the string being checked for quotes\n    :return: True if a string\
    \ is quoted\n    \"\"\"\n    return len(arg) > 1 and arg[0] == arg[-1] and arg[0]\
    \ in constants.QUOTES"
- source_sentence: python popen git tag
  sentences:
  - "def git_tag(tag):\n    \"\"\"Tags the current version.\"\"\"\n    print('Tagging\
    \ \"{}\"'.format(tag))\n    msg = '\"Released version {}\"'.format(tag)\n    Popen(['git',\
    \ 'tag', '-s', '-m', msg, tag]).wait()"
  - "def find_centroid(region):\n    \"\"\"\n    Finds an approximate centroid for\
    \ a region that is within the region.\n    \n    Parameters\n    ----------\n\
    \    region : np.ndarray(shape=(m, n), dtype='bool')\n        mask of the region.\n\
    \n    Returns\n    -------\n    i, j : tuple(int, int)\n        2d index within\
    \ the region nearest the center of mass.\n    \"\"\"\n\n    x, y = center_of_mass(region)\n\
    \    w = np.argwhere(region)\n    i, j = w[np.argmin(np.linalg.norm(w - (x, y),\
    \ axis=1))]\n    return i, j"
  - "def zero_pad(m, n=1):\n    \"\"\"Pad a matrix with zeros, on all sides.\"\"\"\
    \n    return np.pad(m, (n, n), mode='constant', constant_values=[0])"
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'python popen git tag',
    'def git_tag(tag):\n    """Tags the current version."""\n    print(\'Tagging "{}"\'.format(tag))\n    msg = \'"Released version {}"\'.format(tag)\n    Popen([\'git\', \'tag\', \'-s\', \'-m\', msg, tag]).wait()',
    'def zero_pad(m, n=1):\n    """Pad a matrix with zeros, on all sides."""\n    return np.pad(m, (n, n), mode=\'constant\', constant_values=[0])',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.6932, -0.0514],
#         [ 0.6932,  1.0000, -0.0655],
#         [-0.0514, -0.0655,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 19,604 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                          |
  |:--------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                              |
  | details | <ul><li>min: 6 tokens</li><li>mean: 9.88 tokens</li><li>max: 23 tokens</li></ul> | <ul><li>min: 36 tokens</li><li>mean: 86.93 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  |:--------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>how similar is java to python</code>                                | <code>def java_version():<br>    """Call java and return version information.<br><br>    :return unicode: Java version string<br>    """<br>    result = subprocess.check_output(<br>        [c.JAVA, '-version'], stderr=subprocess.STDOUT<br>    )<br>    first_line = result.splitlines()[0]<br>    return first_line.decode()</code>                                                                                                         |
  | <code>how to add values to display on top of each bar chart python</code> | <code>def _change_height(self, ax, new_value):<br>        """Make bars in horizontal bar chart thinner"""<br>        for patch in ax.patches:<br>            current_height = patch.get_height()<br>            diff = current_height - new_value<br><br>            # we change the bar height<br>            patch.set_height(new_value)<br><br>            # we recenter the bar<br>            patch.set_y(patch.get_y() + diff * .5)</code> |
  | <code>python makedirs if not exist</code>                                 | <code>def makedirs(path):<br>    """<br>    Create directories if they do not exist, otherwise do nothing.<br><br>    Return path for convenience<br>    """<br>    if not os.path.isdir(path):<br>        os.makedirs(path)<br>    return path</code>                                                                                                                                                                                           |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.8157 | 500  | 0.1485        |


### Framework Versions
- Python: 3.12.7
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.0+cpu
- Accelerate: 1.11.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->