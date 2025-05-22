from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter

@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, "r", encoding="utf-8") as f:
        xml_content = f.read().replace('&', '&amp;')
    root = ET.fromstring(xml_content)
    sentence_pairs = []
    alignments = []
    for sentence in root.findall(".//s"):
        source_elem = sentence.find("english")
        target_elem = sentence.find("czech")
        source_text = source_elem.text.strip().split()
        target_text = target_elem.text.strip().split()
        sentence_pairs.append(SentencePair(source=source_text, target=target_text))
        sure_alignments = []
        possible_alignments = []
        alignment_elem_sure = sentence.find("sure")
        alignment_elem_possible = sentence.find("possible")
        if (((alignment_elem_sure is not None) and (alignment_elem_sure.text is not None))):
            sure_alignments = [
                tuple(map(int, pair.split('-')))
                for pair in alignment_elem_sure.text.strip().split()]
        if (((alignment_elem_possible) is not None and (alignment_elem_possible.text is not None))):
            possible_alignments = [
                tuple(map(int, pair.split('-')))
                for pair in alignment_elem_possible.text.strip().split()]
        alignments.append(LabeledAlignment(sure=sure_alignments, possible=possible_alignments))
    return sentence_pairs, alignments

def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    source_counter = Counter()
    target_counter = Counter()
    for pair in sentence_pairs:
        source_counter.update(pair.source)
        target_counter.update(pair.target)
    def filter_vocab(counter, cutoff):
        sorted_vocab = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if cutoff is not None:
            sorted_vocab = sorted_vocab[:cutoff]
        return [word for word, _ in sorted_vocab]
    source_vocab = filter_vocab(source_counter, freq_cutoff)
    target_vocab = filter_vocab(target_counter, freq_cutoff)
    source_dict = {word: idx for idx, word in enumerate(source_vocab)}
    target_dict = {word: idx for idx, word in enumerate(target_vocab)}
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.

    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    for pair in sentence_pairs:
        source_tokens = [source_dict[token] for token in pair.source if token in source_dict]
        target_tokens = [target_dict[token] for token in pair.target if token in target_dict]
        if not source_tokens or not target_tokens:
            continue
        tokenized_sentence_pairs.append(
            TokenizedSentencePair(
                source_tokens=np.array(source_tokens, dtype=np.int32),
                target_tokens=np.array(target_tokens, dtype=np.int32)))
    return tokenized_sentence_pairs
