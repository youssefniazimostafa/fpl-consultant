# CLAUDE.md

## Final report (standing rule)

At the end of every task, without being asked, finish your response with the FINAL REPORT in the structure below.

When you have finished all the work, immediately write this report as the last part of your response. Do not wait to be asked for it, and do not stop with only a short status message. Use this exact structure, so it can be pasted to a reviewer:

1. Summary: 2-3 sentences on what was done and whether it fully worked.
2. Changes: each commit hash with a one-line description, and the files touched.
3. Results: every number or test outcome the task asked for, in a small table where it helps.
4. Problems and deviations: anything that failed, anything you did differently from the instructions and why, and any assumptions you made.
5. Needs my decision: open questions or recommended next steps, or "None".
6. State: output of `git status` and `git log --oneline -5`, and whether everything is pushed.

Be factual: don't claim something was tested or evaluated unless you actually ran it.

## Scheduled runs (cron backup)

Two cron-job.org jobs trigger the workflow through `workflow_dispatch`:

- **"FPL pre-deadline check"**: hourly at :45, with input `pre_deadline_check` = true
  (goes through the deadline gate, so only one run freezes each recommendation);
- **"FPL full refresh"**: at 06:00 and 18:00, with no inputs (a normal full run).

Both use a fine-grained GitHub token limited to this repo's Actions
(read and write). **It expires on 15 June 2027 and must be renewed then**
(create a new one with the same settings and paste it into both jobs).
Never put the token in the repo.

## Reinstalling the iPhone app every 7 days

The iPhone app (Tauri iOS, viewer mode only) is signed with a free Apple ID,
so it stops opening after 7 days. To renew: plug the iPhone into the Mac,
unlock it, then run

```
cd "/Users/youssef/Desktop/Graduate Classes and Materials/PROGRAMMING FOR BUSINESS /Online Website Repo/fpl-consultant/app"
./scripts/iphone-install.sh
```

and keep the phone unlocked until the app opens. If the script says no
signing certificate was found: Xcode > Settings > Accounts > sign in with
the Apple ID > select "(Personal Team)" > Manage Certificates > + > Apple
Development, then run it again. One-time setup and the web-app alternative
are in README.md ("On your iPhone"). Never put the Apple ID, team ID or any
signing files in the repo; the script reads the team ID from the login
keychain at build time.
