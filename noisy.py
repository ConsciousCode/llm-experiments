import random
from torch.utils.data import Dataset

class NoisyTextDataset(Dataset):
    def __init__(self, dataset, p=0.1):
        self.dataset = dataset
        self.p = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, label = self.dataset[idx]
        sample = self.apply_noise(sample)
        return sample, label

    def apply_noise(self, text):
        num_ops = int(len(text) * self.p)
        positions = random.sample(range(len(text)), num_ops)
        result = []
        prev = 0

        for pos in sorted(positions):
            result.append(text[prev:pos])
            ins, prev = self.char_op(text, pos)
            if ins:
                result.append(ins)
        
        if text := text[prev:]:
            result.append(text)
        
        return ''.join(result)

    def char_op(self, text, pos):
        op = random.choice(['insert', 'delete', 'transpose', 'replace', 'typo'])

        match op:
            case 'insert': return chr(random.randint(0x110000)), pos
            case 'delete': pos += 1
            case 'transpose':
                if pos + 1 < len(text):
                    return text[:2][::-1], pos + 2
            case 'replace': return random.choice(self.char_pool), pos + 1
            case 'typo':
                if text[0] in self.typo_map:
                    return random.choice(self.typo_map[text[0]]), pos + 1
        
        # Default to no-op
        return '', pos