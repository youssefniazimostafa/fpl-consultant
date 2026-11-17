// The "?" help on each UCL card (same idea as src/help.ts for the PL pages).
const XP =
  "Expected points (xP) are our estimate of a player's points this matchday, from his chance of playing, his minutes, his goal threat, the opponent and UEFA's scoring rules.";

export const UCL_HELP: Record<string, string> = {
  "Recommended squad": `The 15 players to pick, within the budget and the club limit, with the day each one plays. ${XP}`,
  "Captain": "The captain's points are doubled. A captain who plays on the first match day can be switched to a later-day player if he scores little, so that option counts when we choose.",
  "Transfers and plan": "The plan up to the next unlimited-transfer window: transfers each matchday, free transfers, any -4 hits. Only this matchday's moves are final.",
  "Chips": "Wildcard (unlimited transfers, kept) and Limitless (one matchday with no budget limit, then your team returns). Each once, not on MD1, MD9 or MD11.",
  "Match days": "Every matchday is played over two or three days. Players who play later can still be brought in between days.",
  "How the model works": "What XPert's expected points are built from, in plain words. Every part was tested on past matchdays before it was switched on.",
  "Why these players": "One plain reason per pick: the fixture, the day, how likely he is to start and where his points should come from.",
  "Between days": "After the first match day you may make up to 4 substitutions and switch the captain. Any change turns off the automatic substitutions, so we only suggest it when it clearly gains points.",
  "Value of flexibility": "How many points, on average, the chance to change things between days adds to this squad. Players on later days make it bigger.",
  "Last matchday": "How the recommended team did. Before MD5 there was no live recommendation, so it's the team XPert would have picked from the data before that matchday (a replay).",
  "Best possible": "The best team anyone could have picked, knowing the results. Nobody can know them in advance, so it's the ceiling, not a fair target.",
  "Top expected points": `The players expected to score most this matchday. ${XP}`,
  "Best captains": "Expected points doubled: who should score most with the armband.",
  "Best value": "Expected points for every €1m of price.",
  "In form": "Points over the last two matchdays.",
  "Differentials": "Good options picked by fewer than 5% of managers: they gain you ground on the average.",
  "Most selected": "The players most managers have.",
  "Injury and team news": "UEFA's own status: injured, doubtful or suspended.",
  "Matchday plan": "XPert's plan matchday by matchday up to the next unlimited-transfer window: each planned move with the expected points it gains over the rest of the plan, against the bar a move must clear (0.5 points, 4.5 if it costs a -4 hit).",
  "Matchday calendar": "All 17 matchdays: the stage, the days they're played, free transfers (2 in the league phase with 1 carried over; unlimited before MD1, MD9 and MD11; 2, 3, 5, 3, 5, 3, 5 in the knockouts with no carry-over), the club limit, the budget and whether chips can be played.",
  "Club fixtures": "Every club's coming matches, the day each is played and how hard it is by our team ratings (green easy, red hard), easiest run first, plus each club's chance of the top 8.",
  "Your plan for": "Enter your own changes, captain, vice-captain and chip for the next matchday. They're checked against the UCL rules, saved only on this Mac, and the Live consultant compares them with XPert's plan: strengths, weaknesses, points and the risks of each plan. The choice stays yours.",
  "Points vs the average": "The recommended team's points each matchday against an estimate of the game average, and the best possible team.",
  "Backtest": "We replayed every played matchday, predicting it only from the matches before it, and compared with picking by points per game so far.",
  "Knockout chances": "From thousands of simulated seasons: each club's chance to finish in the top 8 (straight to the round of 16), 9th-24th (play-offs) or out, and how many matches it should have left.",
  "When XPert updates": "Runs are timed from the fixture list: the evening before, 2 hours before the deadline, after each match day and the next morning.",
};

export function uclHelp(title: unknown): string | undefined {
  if (typeof title !== "string") return undefined;
  return UCL_HELP[title.replace(/,?\s*MD\d+\b/g, "").trim()];
}
