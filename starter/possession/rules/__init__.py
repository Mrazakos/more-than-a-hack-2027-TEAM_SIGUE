"""The rules, one module per group. Numbers are historical (3, 10 and 12 were removed): never renumber.

    rule 0      keeper slots              tracking.py / context.py   (the *_keeper features)
    rule 1      kickoff                   kickoff.py
    rule 2, 11  touch, backdating, pass-by  touch.py (who touches the ball) + pass_loop.py (backdating)
    rules 4-6   ball out, corner, goal kick, throw-in, goal   restarts.py
    rules 7, 8, 8b  free kicks, untagged stoppages            stoppages.py
    rule 13     stoppage lead             pass_loop.py
    rules 15, 14, 9  shape veto, dead tail, merge  (run in this order)   postprocess.py

The flow (see labeller.label_half): the event tables above are computed first (restarts,
stoppages, slow stops), then pass_loop.run_pass walks the half frame by frame, then the
postprocess passes tidy the labels.
"""
