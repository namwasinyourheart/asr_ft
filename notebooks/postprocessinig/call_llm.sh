curl --location 'https://ai.vnpost.vn/vllm-openai-oss-143/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "openai/gpt-oss-20b",
    "messages": [
        {"role": "system", "content": "Bạn là một chuyên gia của Tổng công ty Bưu điện Việt Nam."},
        {"role": "user", "content": "Bạn là ai"}
    ]
}'
