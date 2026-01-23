"""
Tokenization实验
学习目标: 理解不同tokenizer的行为
"""

from transformers import (GPT2Tokenizer, BertTokenizer, AutoTokenizer)

def explore_tokenization():
    #1. GPT-2 Tokenizer (BPE):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    #2. BERT Tokenizer (wordpiece):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    texts = [
        "Hello, how are you?",
        "人工智能真厉害！",
        "print('Hello World')",
        "supercalifragilisticexpialidocious",
        "The quick brown fox jumps over the lazy dog."
    ]

    print("="*60)
    print("TOKENIZATION比较")
    print("="*60)

    for text in texts:
        print(f"\n原文: {text}")
        print("-"*60)

        gpt2_tokens = gpt2_tokenizer.tokenize(text)
        gpt2_ids = gpt2_tokenizer.encode(text)
        print(f"GPT-2 tokens ({len(gpt2_tokens)}): {gpt2_tokens}")
        print(f"GPT-2 IDs: {gpt2_ids}")

        bert_tokens = bert_tokenizer.tokenize(text)
        bert_ids = bert_tokenizer.encode(text)
        print(f"BERT tokens ({len(bert_tokens)}): {bert_tokens}")
        print(f"BERT IDs: {bert_ids}")

        decoded_gpt2  = gpt2_tokenizer.decode(gpt2_ids)
        print(f"Decoded: {decoded_gpt2}")

        decoded_bert = bert_tokenizer.decode(bert_ids)
        print(f"Decoded: {decoded_bert}")


def understand_special_tokens():
    """理解特殊tokens: [CLS], [SEP], [PAD], [UNK]"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("\n特殊Tokens:")
    print(f"[CLS] token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"[SEP] token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"[PAD] token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"[UNK] token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

    # 实际使用示例
    text = "Hello world"
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # 添加[CLS]和[SEP]
        max_length=10,
        padding='max_length',  # 填充到max_length
        return_tensors='pt'  # 返回PyTorch tensor
    )

    print(f"\n编码后的序列: {encoded['input_ids']}")
    print(f"Attention mask: {encoded['attention_mask']}")


def token_limits_experiment():
    """实验context window限制"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # GPT-2的context window是1024 tokens
    long_text = "AI is amazing! " * 200  # 故意创建很长文本

    tokens = tokenizer.encode(long_text)
    print(f"\n总token数: {len(tokens)}")
    print(f"GPT-2 限制: 1024 tokens")

    if len(tokens) > 1024:
        print(f"超出限制! 需要截断或分块处理")
        # 截断策略
        truncated = tokens[:1024]
        decoded = tokenizer.decode(truncated)
        print(f"截断后文本前50字符: {decoded[:50]}...")


if __name__ == "__main__":
    explore_tokenization()
    understand_special_tokens()
    token_limits_experiment()
