# LinkedIn draft — mid-season update

*(~200 words, drafted after GW9/Step 11; edit freely before posting.)*

---

Nine gameweeks into Fantasy Premier League, I've been building something a bit different from the usual
spreadsheet: a small end-to-end data pipeline that pulls live FPL data, blends it with betting odds, team-strength
ratings and underlying attacking stats, and turns all of it into a weekly expected-points model for every player
in the game.

The fun part isn't the prediction — it's the feedback loop. Every week, the system scores last week's own
predictions against what actually happened, splits the error into minutes, attacking returns, clean sheets and
bonus, and tells me exactly where the model is weakest (currently: predicting who actually starts). That log
compounds into an accuracy track record I can point to, not just vibes.

From there it's a transfer/captain optimiser (a mixed-integer program, not a greedy guess), a league-aware layer
that tracks my 9 rivals' ownership so I'm not just chasing average points but real rank, and now a Monte Carlo
simulation that estimates my odds of actually finishing top of our mini-league each week — sometimes a different
answer than "pick the highest expected score."

Currently 2nd in a 10-manager league, 9 points off top. Small hobby project, real engineering: data pipelines,
optimisation, and the discipline of actually measuring whether a model is working.

#FPL #DataScience #SideProject
