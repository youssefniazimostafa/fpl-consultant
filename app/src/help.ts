// The "?" help on each main card: two or three plain sentences on what the
// card shows, how to read it and where the numbers come from. Keyed by the
// card's title with any gameweek number taken out ("Top projected points,
// GW11" -> "Top projected points"), so one entry covers every week.

const XP = "Expected points (xP) are our estimate of what a player will score this gameweek, from his chance of playing, his minutes, his goal threat and the fixture.";

export const HELP: Record<string, string> = {
  // Recommended team
  "Transfers": "The transfer to make this gameweek, if any. A move is only suggested when it beats saving the free transfer by a clear margin. A hit costs 4 points per extra transfer.",
  "Captain": "Who to give the armband, and the vice-captain who takes over if he doesn't play. When two players are close, we prefer the one fewer of your rivals own.",
  "Chips": "Which chips you still have and the best week to play each one, from how many extra points it should bring.",
  "Why this team": "The reasons behind this week's team, transfer and captain, in plain words.",
  "You vs the consultant": "Your real points each week against the team we recommended before the deadline, scored with FPL's own rules.",
  // Last gameweek
  "Last gameweek": "How your team did last gameweek, player by player, against what we expected.",
  "Points that counted": "Your starting XI's points, with the captain's doubled. Bench points don't count unless a substitute came on.",
  "Best you could have done": "The best team you could have picked from your own squad, knowing the results. It shows how much was left on the bench or the armband.",
  "Where every point came from": "Your points split into appearances, goals, assists, clean sheets, bonus and the rest, so you can see what really scored.",
  "How predictable was it?": "How close our expected points were to what players really scored last gameweek.",
  "In short": "The main points of the gameweek in a few lines.",
  // Team of the week
  "Team of the week": "The highest-scoring XI of the gameweek across all of FPL, within the usual formation rules.",
  // Your leagues
  "Your leagues": "Your mini-league: the table, your rivals' teams and who owns what.",
  "Standings": "Your league table after the last gameweek. Tap a team to see its latest confirmed line-up.",
  "Rivals in your area": "The rivals within about a gameweek's points of you, the transfer they're most likely to make next, their free transfers and the chips they have left.",
  "Who owns what": "The players most of your rivals own. If you don't have one and he scores big, you lose ground.",
  "Popular players you don't own": "Players many rivals own that you don't: the ones whose big week would hurt you most.",
  "Differentials": "Good players few or none of your rivals own. They gain you ground when they score.",
  // Scout
  "Top expected points": `The players expected to score the most next gameweek. ${XP}`,
  "Best captain options": "The players who'd score the most with the armband next gameweek (their expected points doubled).",
  "Slept-on players": "Players who scored far more than we expected last gameweek. Doing it week after week can mean a player is finding form.",
  "In form lately": "Players scoring clearly more per game over the last few gameweeks than earlier in the season.",
  "Cheap high scorers": "Points scored for every £1m of price: the best value in the game right now.",
  "Injury and team news": "FPL's own injury and suspension news from the last week, newest first, with the chance of playing.",
  // Planner
  "Your plan": "Pick your own transfers, captain and chip. It's checked against FPL's rules as you go, saved only on this Mac, and the consultant compares it with its own plan.",
  "Planner": "The plan for the next few gameweeks: transfers, free transfers and the best week for each chip. Only this week's moves are final.",
  "Your players' fixtures": "Your clubs' next games, coloured by FPL's own difficulty rating: green is easy, pink is hard.",
  // Analytics
  "Points per gameweek": "Your points each gameweek against your league's average, the leader and the average of every FPL manager.",
  "Your league position": "Where you stood in your league after each gameweek (1st at the top).",
  "Who owns what in your league": "The most-owned players in your league and how each manager used them last gameweek: starting, captain, benched or not owned.",
  "Expected vs actual": "Your team's expected points before each deadline against what it really scored.",
  "Biggest misses by gameweek": "The players our expected points got most wrong, so you can see what the numbers can't predict, like hauls and surprise starts.",
  "Fixture difficulty, next 6 gameweeks": "Every club's next six games by FPL's own difficulty rating, easiest run at the top.",
  "Our predictions vs FPL's own": "How close our expected points and FPL's own predictions were to the real scores, gameweek by gameweek.",
  "Transfer or hold: chances this week": "How often each option tops your league this gameweek in thousands of simulated weeks. The green bar is the recommendation.",
  "Chances of being top by GW19": "Each manager's chance of leading your league at the halfway point, from simulating the rest of the first half.",
  "Captain board": "Your starters' expected points this gameweek. Green is the recommended captain.",
  "Your options this week": "The main captain and transfer choices side by side: what your team should score, a realistic bad and good week, and how each plays against your rivals.",
  "Backtest: how the model did on last season": "We replayed last season week by week, predicting each gameweek only from what was known before it, and compared the results with simple methods.",
};

/** The help for a card title, ignoring gameweek numbers in it. */
export function helpFor(title: unknown): string | undefined {
  if (typeof title !== "string") return undefined;
  const key = title
    .replace(/,?\s*(for\s+)?GW\d+\b/gi, "")
    .replace(/\s+/g, " ")
    .trim();
  return HELP[key] ?? HELP[title];
}
