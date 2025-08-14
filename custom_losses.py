import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class MetricalLoss(nn.Module):
    def __init__(self, alpha=0.9, beta=0.1, syllable_weight=1.0):
        super().__init__()
        self.alpha = alpha  # weight for language modeling loss
        self.beta = beta    # weight for meter consistency loss
        self.syllable_weight = syllable_weight
        
    def count_syllables_simple(self, text):
        """Simple syllable counting heuristic"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        total_syllables = 0
        for word in words:
            # Simple heuristic: count vowel groups
            vowels = 'aeiouy'
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Every word has at least 1 syllable
            syllable_count = max(1, syllable_count)
            total_syllables += syllable_count
            
        return total_syllables
        
    def forward(self, logits, targets, tokenizer=None):
        # Standard language modeling loss
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        # If no tokenizer provided, just return standard loss
        if tokenizer is None:
            return lm_loss
            
        # Decode predictions and compute meter penalty
        with torch.no_grad():
            predicted_ids = torch.argmax(logits, dim=-1)
            meter_penalty = 0.0
            
            for i in range(predicted_ids.size(0)):  # batch dimension
                try:
                    # Decode the sequence
                    decoded_text = tokenizer.decode(predicted_ids[i].cpu().numpy())
                    lines = decoded_text.split('\n')
                    
                    line_penalties = []
                    for line in lines:
                        line = line.strip()
                        if len(line) > 0:
                            syllables = self.count_syllables_simple(line)
                            # Penalize deviation from 10 syllables (iambic pentameter)
                            deviation = abs(syllables - 10)
                            line_penalties.append(deviation * self.syllable_weight)
                    
                    if line_penalties:
                        meter_penalty += sum(line_penalties) / len(line_penalties)
                        
                except:
                    # If decoding fails, no penalty
                    pass
            
            meter_penalty = meter_penalty / predicted_ids.size(0) if predicted_ids.size(0) > 0 else 0.0
        
        # Combine losses
        total_loss = self.alpha * lm_loss + self.beta * meter_penalty
        return total_loss

class RhymeAwareLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.15):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def simple_rhyme_score(self, word1, word2):
        """Simple rhyme scoring based on suffix similarity"""
        if len(word1) < 2 or len(word2) < 2:
            return 0.0
            
        # Check last 2-3 characters for rhyme
        suffix1 = word1[-3:] if len(word1) >= 3 else word1[-2:]
        suffix2 = word2[-3:] if len(word2) >= 3 else word2[-2:]
        
        if suffix1 == suffix2:
            return 1.0
        elif word1[-2:] == word2[-2:]:
            return 0.7
        elif word1[-1] == word2[-1]:
            return 0.3
        return 0.0
        
    def forward(self, logits, targets, tokenizer=None):
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        if tokenizer is None:
            return lm_loss
            
        # Compute rhyme consistency bonus
        with torch.no_grad():
            predicted_ids = torch.argmax(logits, dim=-1)
            rhyme_bonus = 0.0
            
            for i in range(predicted_ids.size(0)):
                try:
                    decoded_text = tokenizer.decode(predicted_ids[i].cpu().numpy())
                    lines = [line.strip() for line in decoded_text.split('\n') if line.strip()]
                    
                    if len(lines) >= 2:
                        # Check rhyme scheme (simple ABAB pattern)
                        rhyme_score = 0.0
                        valid_pairs = 0
                        
                        for j in range(0, len(lines)-1, 2):  # Every other line
                            if j+1 < len(lines):
                                # Get last word of each line
                                words1 = lines[j].split()
                                words2 = lines[j+1].split()
                                
                                if words1 and words2:
                                    last_word1 = re.sub(r'[^\w]', '', words1[-1].lower())
                                    last_word2 = re.sub(r'[^\w]', '', words2[-1].lower())
                                    
                                    rhyme_score += self.simple_rhyme_score(last_word1, last_word2)
                                    valid_pairs += 1
                        
                        if valid_pairs > 0:
                            rhyme_bonus += rhyme_score / valid_pairs
                except:
                    pass
            
            rhyme_bonus = rhyme_bonus / predicted_ids.size(0) if predicted_ids.size(0) > 0 else 0.0
        
        # Subtract rhyme bonus from loss (reward good rhyming)
        total_loss = self.alpha * lm_loss - self.beta * rhyme_bonus
        return total_loss