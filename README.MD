# model 
- mixed effect model
X: rest of the cols
Y: probing types and level of depth

# research questions
- how rest of the cols impact on the Y
- which type of probing leads to deeper insights

# data types
| Column Name           | Current Python Type | Desired Python Type |
| --------------------- | ------------------- | ------------------- |
| Start                 | str                 | timestamp           |
| End                   | str                 | timestamp           |
| Speaker               | str                 | str                 |
| Transcript            | str                 | str                 |
| Mapped Q#             | str                 | str                 |
| Goal Mapping          | str                 | list[str]           |
| Goal Relevant?        | str                 | bool                |
| Purpose if Irrelevant | str                 | list[str]           |
| Theme(s)              | str                 | list[str]           |
| Probing Type          | str                 | list[str]           |
| Depth                 | str                 | float               |


# plots
Goal Mapping:
    str (parsed as list of strs) e.g., ['Bonding', 'Closing', 'Habits/Routine', 'Introduction/Bio', 'Motivations/Barriers', 'Support Systems', 'Tech Solutions/Support Systems'] do not seprate treat each string as an entity
    separate by human vs ai

power post-poc

[x] full mixed effect model

# note
Mapped Q#:
    str (e.g., ['Closing', 'Q0', 'Q1', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Warm-Up Q'])

Goal Mapping:
    str (parsed as list of strs) e.g., ['Bonding', 'Closing', 'Habits/Routine', 'Introduction/Bio', 'Motivations/Barriers', 'Support Systems', 'Tech Solutions/Support Systems'] using separator "/"

Goal Relevant?
    str needs to be parsed as bool ("Yes" or "No")

[x] derive:
    overall session length (by word count)
