import pandas as pd
from anthropic import Anthropic
import time
from typing import List, Tuple

def process_narratives(
    file_path: str,
    api_key: str,
    narrative_prompt: str,
    comparison_prompt: str
) -> None:
    """
    Process narratives using Claude API and save results to the same TSV file.
    
    Args:
        file_path: Path to the TSV file
        api_key: Anthropic API key
        narrative_prompt: Prompt for processing individual narratives
        comparison_prompt: Prompt for comparing narrative pairs
    """
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Read TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Process individual narratives
    df['Narrative_roles_and_salient_elements'] = df['Narrative'].apply(
        lambda x: process_single_narrative(x, client, narrative_prompt)
    )
    
    # Process narrative pairs
    df['Elements_persistence'] = process_narrative_pairs(
        df['Narrative_roles_and_salient_elements'].tolist(),
        client,
        comparison_prompt
    )
    
    # Save results
    df.to_csv(file_path, sep='\t', index=False)

def process_single_narrative(
    narrative: str,
    client: Anthropic,
    prompt: str,
    max_retries: int = 3,
    delay: int = 1
) -> str:
    """Process a single narrative using Claude API with retry logic."""
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"{prompt}\n\nNarrative: {narrative}"
                }]
            )
            return message.content
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            time.sleep(delay * (attempt + 1))

def process_narrative_pairs(
    narratives: List[str],
    client: Anthropic,
    prompt: str
) -> List[str]:
    """Process consecutive pairs of narratives."""
    results = [''] * len(narratives)  # Initialize with empty strings
    
    for i in range(len(narratives) - 1):
        pair_result = process_single_narrative(
            f"First: {narratives[i]}\nSecond: {narratives[i+1]}",
            client,
            prompt
        )
        results[i+1] = pair_result
    
    return results

# Example usage:
if __name__ == "__main__":
    FILE_PATH = "/Users/stefanodegiorgis/Desktop/AI-Film-Festival/female-scientist.tsv"
    API_KEY = ""
    
    NARRATIVE_PROMPT = """
    You receive a piece of text with a small story.
    Identify in the piece of text the salient elements which constitutes relevant narrative roles, for example:
    Story: "The dog was sleeping when the bird hit him on the nose. A cat arrived to help the poor dog and together got rid
    of the evil bird."
    {
    "Characters": {
    "Main Character" : "Dog",
    "Helping Character" : "Cat",
    "Antagonist" : "Cat"
    }
    }

    Introduce new roles accordingly to the elements that you identify as relevant in the story, the important is that you keep
    track of all the participants which are relevant for the small story.
    In addition to this, focus on salient events and relevant moments in the narrative, e.g.
    {
    "Events": {
        "Event1": "Aggression by the Bird against the Dog",
        "Event2": "Help provided by the Cat to the Dog",
        "Event3": "Fight between the Dog and the Cat vs the Bird",
        "Event4": "Defeat of the Bird"
    }
    }

    Avoid comments which are not part of the json templates. Print only the annotations. 
    """
    
    COMPARISON_PROMPT = """
    You are given two sets of annotations in json like format.
    Your task is to check both the annotations, and if there are overlaps report them in the output. Note that the overlap
    could be perfect, e.g.
    "Hero" : "Dog"
    "Hero" : "Dog"
    or could be fuzzy, like
    "Hero" : "Dog"
    "Protagonist" : "Black Dog".

    You have to check for these overlaps and align them despite their imperfect overlap, they just have to point at the
    same entities, maybe expressed in a slightly different way.
    For entities which are in "First" but not in "Second" store them in json like structure such as:
    { "Previous" : { ...} }
    For entities which are in both "First" and "Second" store them in json like structure such as:
    { "Persistent" : { ...} }
    For entities which are not in "First" but are introduced in "Second" store them in json like structure such as:
    { "Newly Introduced" : { ...} }
    Avoid comments which are not part of the json templates. Print only the annotations. 
    """
    
process_narratives(FILE_PATH, API_KEY, NARRATIVE_PROMPT, COMPARISON_PROMPT)