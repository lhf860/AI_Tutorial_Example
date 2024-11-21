# generate函数说明

"""
类用于保存生成任务的配置。调用generate支持以下对于text-decoder、text-to-text、speech-to-text和vision-to-text模型的生成方法：

如果num_beams=1且do_sample=False，则使用贪婪搜索，调用~generation.GenerationMixin.greedy_search。
如果penalty_alpha>0且top_k>1，则使用对比搜索，调用~generation.GenerationMixin.contrastive_search。
如果num_beams=1且do_sample=True，则使用多概率采样，调用~generation.GenerationMixin.sample。
如果num_beams>1且do_sample=False，则使用beam搜索，调用~generation.GenerationMixin.beam_search。
如果num_beams>1且do_sample=True，则使用beam搜索多概率采样，调用~generation.GenerationMixin.beam_sample。
如果num_beams>1且num_beam_groups>1，则使用分群束搜索，调用~generation.GenerationMixin.group_beam_search。
如果num_beams>1且constraints!=None或force_words_ids!=None，则使用约束束搜索，调用~generation.GenerationMixin.constrained_beam_search。

在使用这个模型进行文本生成时，您也可以不直接调用上述方法。而是将自定义参数值传递给'generate'方法。

参数说明：

    max_length：控制生成输出的长度，默认为 20。它的值对应于输入提示的长度加上max_new_tokens。如果同时设置了max_new_tokens，则它的效果将被覆盖。
    max_new_tokens：控制要生成的令牌数量，忽略提示中的令牌数量。它的值默认为 0。
    min_length：控制生成序列的最小长度，默认为 0。它的值对应于输入提示的长度加上min_new_tokens。如果同时设置了min_new_tokens，则它的效果将被覆盖。
    min_new_tokens：控制要生成的令牌数量，忽略提示中的令牌数量。它的值默认为 0。
    early_stopping：控制基于 beam 的方法（如 beam-search）的停止条件。它接受以下值：True，表示生成在有num_beams个完整候选项时停止；False，表示应用启发式方法，在找到更好候选项的可能性很小时停止；"never"，表示 beam 搜索过程仅在无法找到更好候选项时停止（经典 beam 搜索算法）。
    max_time：允许计算运行的最大时间，单位为秒。如果分配的时间已过，生成过程仍会完成当前迭代。

这个注释是用于控制生成策略的参数。它包含了以下几个参数：

do_sample（可选，默认为False）：是否使用采样；否则使用贪婪解码。
num_beams（可选，默认为1）：束搜索的束数。1表示不使用束搜索。
num_beam_groups（可选，默认为1）：将num_beams分成若干组，以确保不同束组的多样性。更多详细信息请参考这篇论文(This Paper)。
penalty_alpha（可选）：在对比搜索解码中，平衡模型置信度和退化惩罚的值。
use_cache（可选，默认为True）：模型是否应使用过去的最后一个键/值注意力（如果适用于模型）来加速解码。
"""





在解释这些参数之前，让我们先了解一下这些参数在模型输出 logits（未归一化的概率）的操作中的作用。

temperature (浮点数，可选，默认为 1.0)：

这个值用于调整下一个令牌的概率。通过改变这个值，你可以控制生成的文本的随机性。较大的 temperature 值会导致生成的文本更加随机，而较小的 temperature 值则会生成更加确定性的文本。

top_k (整数，可选，默认为 50)：

这个参数决定了在 top-k 过滤中保留的最高概率词汇令牌的数量。top-k 过滤是一种技术，用于在生成过程中过滤掉不太可能的令牌。

top_p (浮点数，可选，默认为 1.0)：

如果设置为小于 1 的浮点数，那么只有最可能的令牌集合，其概率之和达到或超过 top_p，才会在生成过程中保留。

typical_p (浮点数，可选，默认为 1.0)：

局部典型性衡量在给定部分文本生成条件下，预测下一个令牌的概率与随机预测下一个令牌的概率的相似程度。如果设置为小于 1 的浮点数，那么只有最局部典型的令牌集合，其概率之和达到或超过 typical_p，才会在生成过程中保留。

epsilon_cutoff (浮点数，可选，默认为 0.0)：

如果设置为在 0 和 1 之间的浮点数，那么只有条件概率大于 epsilon_cutoff 的令牌才会被采样。这个参数可以用来控制生成过程中令牌的选择。

eta_cutoff (浮点数，可选，默认为 0.0)：

eta 采样是一种局部典型采样和 epsilon 采样的混合。如果设置为在 0 和 1 之间的浮点数，那么一个令牌只有在它大于 eta_cutoff 或 sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))) 时才会被考虑。后者直观上是预期下一个令牌概率，乘以 sqrt(eta_cutoff)。有关更多详细信息，请参阅 Truncation Sampling as Language Model Desmoothing。

diversity_penalty (浮点数，可选，默认为 0.0)：

如果生成的某个时间点的令牌与同一组其他束的令牌相同，将从束的分数中减去 diversity_penalty。请注意，只有当 group beam search 启用时，diversity_penalty 才有效。

repetition_penalty (浮点数，可选，默认为 1.0)：

重复惩罚参数。1.0 表示没有惩罚。有关更多详细信息，请参阅 this paper。

encoder_repetition_penalty (浮点数，可选，默认为 1.0)：

编码器重复惩罚参数。对不是原始输入中的序列施加指数惩罚。1.0 表示没有惩罚。

length_penalty (浮点数，可选，默认为 1.0)：

用于基于束生成的指数惩罚。它作为序列长度的指数使用，进而用于除以序列的分数。因为分数是序列的对数似然（即负数），所以 length_penalty > 0.0 促进较长序列，而 length_penalty < 0.0 鼓励较短序列。

no_repeat_ngram_size (整数，可选，默认为 0)：

如果设置大于 0，那么在生成过程中，不会重复任何长度为 no_repeat_ngram_size 的 n-gram。这个参数主要用于控制生成文本的多样性，避免重复的 n-gram 导致生成的文本过于单一。

bad_words_ids：一个列表，包含不允许生成的 token ID。如果你想获取不应该出现在生成文本中的单词的 token ID，可以使用 tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids。

force_words_ids：一个列表，包含必须生成的 token ID。如果给出的是一个 List[List[int]]，那么它被视为一个简单的必须包含的单词列表，与 bad_words_ids 相反。如果给出的是一个 List[List[List[int]]]，则会触发一个 析构约束，其中可以允许每个单词的不同形式。

renormalize_logits：一个布尔值，表示是否在应用所有 logits 处理器或 warpers（包括自定义的）后归一化 logits。建议将此标志设置为 True，因为搜索算法假定分数 logits 是归一化的，但一些 logits 处理器或 warpers 会破坏归一化。

constraints：一个包含自定义约束的列表，可以添加到生成中，以确保输出在最合适的方式包含由 Constraint 对象定义的某些 token。

forced_bos_token_id：一个整数，表示在 decoder_start_token_id 之后强制生成的第一个 token 的 ID。这对于多语言模型（如 mBART）很有用，因为第一个生成的 token 应该是目标语言的 token。

forced_eos_token_id：当达到 max_length 时强制生成的最后一个 token 的 ID。可以使用一个列表来设置多个 end-of-sequence token。

remove_invalid_values：一个布尔值，表示是否移除模型可能产生的 nan 和 inf 输出，以防止生成方法崩溃。需要注意的是，使用 remove_invalid_values 可能会降低生成速度。

exponential_decay_length_penalty：一个元组，用于在生成一定数量的 token 后添加一个指数增长的长度惩罚。元组应该是 (start_index, decay_factor) 的形式，其中 start_index 表示惩罚开始的位置，decay_factor 表示指数衰减因子。

suppress_tokens：一个列表，包含在生成过程中将被抑制的 token。SupressTokens logit 处理器会将这些 token 的 log 概率设置为 -inf，以便它们不会被采样。

begin_suppress_tokens：一个列表，包含在生成开始时将被抑制的 token。SupressBeginTokens logit 处理器会将这些 token 的 log 概率设置为 -inf，以便它们不会被采样。

forced_decoder_ids：一个列表，包含表示生成索引和 token 索引映射的整数对。例如，[[1, 123]] 表示第二个生成的 token 总是索引为 123 的 token。