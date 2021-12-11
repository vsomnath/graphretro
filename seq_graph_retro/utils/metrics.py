import torch

def get_accuracy_edits(edit_logits, labels):
    accuracy = torch.tensor(0.0)
    for i in range(len(edit_logits)):
        try:
            if torch.argmax(edit_logits[i]).item() == labels[i].item():
                accuracy += 1.0
        except ValueError:
            if torch.argmax(edit_logits[i]).item() == torch.argmax(labels[i]).item():
                accuracy += 1.0
    return accuracy / len(labels)

def get_accuracy_overall(edit_logits, lg_logits, edit_labels, lg_labels, lengths, device='cpu'):
    acc_edits = torch.zeros(len(edit_labels), dtype=torch.long).to(device)
    acc_lg = torch.zeros(len(lg_labels), dtype=torch.long).to(device)

    _, preds = torch.max(lg_logits, dim=-1)

    for i in range(lg_logits.size(0)):
        length = lengths[i]
        result = torch.eq(preds[i][:length], lg_labels[i][:length]).float()
        if int(result.sum().item()) == length:
            acc_lg[i] = 1

        try:
            if torch.argmax(edit_logits[i]).item() == edit_labels[i].item():
                acc_edits[i] = 1
        except ValueError:
            if torch.argmax(edit_logits[i]).item() == torch.argmax(edit_labels[i]).item():
                acc_edits[i] = 1

    acc_overall = (acc_edits & acc_lg).float()
    return torch.mean(acc_overall)


def get_edit_seq_accuracy(seq_edit_logits, seq_labels, seq_mask):
    max_seq_len = seq_mask.shape[0]
    batch_size = seq_mask.shape[1]
    assert len(seq_edit_logits) == max_seq_len
    assert len(seq_labels) == max_seq_len
    assert len(seq_edit_logits[0]) == batch_size
    lengths = seq_mask.sum(dim=0).flatten()

    check_equals = lambda x, y: torch.argmax(x) == torch.argmax(y)

    acc_matrix = torch.stack([torch.stack([(check_equals(seq_edit_logits[idx][bid], seq_labels[idx][bid])).long()
                 for idx in range(max_seq_len)]) for bid in range(batch_size)])
    assert acc_matrix.shape == (batch_size, max_seq_len)
    acc_matrix = acc_matrix.to(seq_mask.device) * seq_mask.t()
    num_correct = acc_matrix.sum(dim=1)
    assert len(num_correct) == batch_size
    accuracy = (num_correct == lengths).float().mean()
    return accuracy


def get_seq_accuracy_overall(seq_edit_logits, lg_logits, seq_labels, lg_labels, lg_lengths, seq_mask):
    acc_edits = torch.zeros(seq_mask.shape[-1], dtype=torch.long).to(seq_mask.device)
    acc_lg = torch.zeros(seq_mask.shape[-1], dtype=torch.long).to(seq_mask.device)

    _, preds = torch.max(lg_logits, dim=-1)
    max_seq_len = len(seq_edit_logits)
    batch_size = seq_mask.shape[1]
    lengths = seq_mask.sum(dim=0).flatten()

    check_equals = lambda x, y: torch.argmax(x) == torch.argmax(y)

    acc_matrix = torch.stack([torch.stack([(check_equals(seq_edit_logits[idx][bid], seq_labels[idx][bid])).long()
                 for idx in range(max_seq_len)]) for bid in range(batch_size)])
    acc_matrix = acc_matrix.to(seq_mask.device) * seq_mask.t()
    num_correct = acc_matrix.sum(dim=1)

    acc_edits = (num_correct == lengths).long()

    for i in range(lg_logits.size(0)):
        length = lg_lengths[i]
        result = torch.eq(preds[i][:length], lg_labels[i][:length]).float()
        if int(result.sum().item()) == length:
            acc_lg[i] = 1

    acc_overall = (acc_edits & acc_lg).float()
    return torch.mean(acc_overall)


def get_accuracy_bin(scores, labels):
    preds = torch.ge(scores, 0).float()
    acc = torch.eq(preds, labels.float()).float()
    return torch.sum(acc) / labels.nelement()


def get_accuracy(scores, labels):
    _,preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()


def get_accuracy_lg(scores, labels, lengths, device='cpu'):
    _, preds = torch.max(scores, dim=-1)
    results = torch.zeros(scores.size(0), dtype=torch.float).to(device)

    for i in range(scores.size(0)):
        length = lengths[i]
        result = torch.eq(preds[i][:length], labels[i][:length]).float()
        if int(result.sum().item()) == length:
            results[i] = 1
    return torch.sum(results) / scores.size(0)
