import evaluate
import numpy as np

def compute_metrics_wrapper(tokenizer, eval_metrics, model_type):

    def compute_metrics_asr(predictions):
        results = {}
        if not eval_metrics:
            return results

        if not len(eval_metrics):
            return results
            
        label_ids = predictions.label_ids
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        pred_ids = predictions.predictions

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pred_ids = np.argmax(pred_ids, axis=-1)
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if "wer" in eval_metrics:
            metric = evaluate.load("wer")
            wer = 100 * metric.compute(predictions=predictions, references=labels)
            results.update({"wer": wer})

        return results

    if model_type == "ASR":
        return compute_metrics_asr
