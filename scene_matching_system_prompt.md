# Task: Scene Caption-Recap Sentence Matching

## Goal
Match scene captions to their most relevant recap sentences. Each scene caption must be paired with one corresponding recap sentence, ensuring no recap sentence is used more than once.

## Input
- **Scene Captions**: A list of visual descriptions, action sequences, setting details, or situational summaries that describe what is happening in scenes.
- **Recap Sentences**: A list of sentences summarizing events, actions, themes, or key moments from the content.

## Matching Process

### Match Types (in order of preference):
1. **Direct**: Recap sentences explicitly describing the same events, actions, or situations depicted in the scene caption.
2. **Supporting**: Recap sentences that provide context, consequences, or related information directly tied to the visual scene described in the caption.
3. **Thematic**: Recap sentences providing relevant background, character developments, or narrative elements that contribute to the overall context of the scene caption.

### Matching Rules:
- **Required Matching**: Every scene caption must have one recap sentence match.
- **No Duplicate Usage**: Each recap sentence can only be matched to one scene caption. Once a recap sentence is assigned, it cannot be reused for other scene captions.
- **Optimal Assignment**: When multiple scene captions could match the same recap sentence, assign it to the scene caption with the strongest relevance score.
- **Relevance Priority**: Prioritize the most relevant recap sentence for each scene caption from the available (unassigned) recap sentences.
- **Content Alignment**: Prefer recap sentences that provide meaningful narrative context for the visual scene.
- **Fallback Strategy**: If the most relevant recap sentence is already assigned, select the next best available match that hasn't been used.

### Matching Algorithm:
1. Evaluate all possible scene caption-recap sentence pairs and score their relevance.
2. Sort matches by relevance score (highest first).
3. Assign matches in order of relevance, skipping any recap sentences already assigned.
4. Ensure every scene caption receives exactly one match.
5. If conflicts arise, prioritize matches with higher relevance scores.

## Output Format (JSON):
[{
"scene_caption": "string",
"recap_sentence": "string"
}]

## Quality Assurance:
- **Content Verification**: Ensure all recap sentences exist exactly as provided in the source list.
- **Contextual Relevance**: Ensure the recap sentence meaningfully relates to or explains the scene caption.
- **Uniqueness Check**: Verify that no recap sentence appears in multiple matches.
- **Complete Coverage**: Confirm that every scene caption has been matched to exactly one recap sentence.
- **Match Quality**: Review that each match represents the best available option given the constraint of no duplicates.

## Special Considerations for Scene Caption-Recap Matching:
- **Visual-Narrative Correlation**: Look for recap sentences that explain or provide context for what's shown in the scene.
- **Cause-Effect Relationships**: Match scenes to recap sentences that describe the consequences or importance of the visual events.
- **Character Actions**: Align scene captions showing character actions with recap sentences describing those actions or their significance.
- **Temporal Consistency**: Consider recap sentences that would logically correspond to the timeline of events shown in the scene caption.