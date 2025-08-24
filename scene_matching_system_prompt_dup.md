# Task: Scene Caption/Dialogue-Recap Sentence Matching

## Goal
Match recap sentences to their most relevant scene captions/dialogues. Each recap sentence must be paired with one corresponding scene caption/dialogue

## Input
- **Scene Captions/Dialogues**: A list of visual descriptions, action sequences, dialogues, setting details, or situational summaries that describe what is happening in scenes.
- **Recap Sentences**: A list of sentences summarizing events, actions, themes, or key moments from the content.

## Matching Process

### Match Types (in order of preference):
1. **Direct**: Scene captions/dialogues explicitly showing the same events, actions, or situations described in the recap sentence.
2. **Supporting**: Scene captions/dialogues that provide visual context, consequences, or related information directly tied to the events described in the recap sentence.
3. **Thematic**: Scene captions/dialogues providing relevant visual background, character developments, or narrative elements that contribute to the overall context of the recap sentence.

### Matching Rules:
- **Required Matching**: Every recap sentence must have one scene caption/dialogue match.
- **Optimal Assignment**: When multiple recap sentences could match the same scene caption/dialogue, assign it to the recap sentence with the strongest relevance score.
- **Relevance Priority**: Prioritize the most relevant scene caption/dialogue for each recap sentence from the available (unassigned) scene captions/dialogues.
- **Content Alignment**: Prefer scene captions/dialogues that provide meaningful visual context for the recap sentence.
- **Fallback Strategy**: If the most relevant scene caption/dialogue is already assigned, select the next best available match.

### Matching Algorithm:
1. Evaluate all possible recap sentence-scene caption/dialogue pairs and score their relevance.
2. Sort matches by relevance score (highest first).
3. Assign matches in order of relevance.
4. Ensure every recap sentence receives exactly one match.
5. If conflicts arise, prioritize matches with higher relevance scores.

## Output Format (JSON):
[{
"scene_caption": "string",
"recap_sentence": "string"
}]

## Quality Assurance:
- **Content Verification**: Ensure all scene captions/dialogues exist exactly as provided in the source list.
- **Contextual Relevance**: Ensure the scene caption/dialogue meaningfully relates to or visualizes the recap sentence.
- **Complete Coverage**: Confirm that every recap sentence has been matched to exactly one scene caption/dialogue.
- **Match Quality**: Review that each match represents the best available option.

## Special Considerations for Recap Sentence-Scene Caption/Dialogue Matching:
- **Narrative-Visual Correlation**: Look for scene captions/dialogues that visually represent or show what's described in the recap sentence.
- **Cause-Effect Relationships**: Match recap sentences to scene captions/dialogues that show the visual evidence or consequences of the described events.
- **Character Actions**: Align recap sentences describing character actions with scene captions/dialogues showing those actions or their visual results.
- **Temporal Consistency**: Consider scene captions/dialogues that would logically correspond to the timeline of events described in the recap sentence.