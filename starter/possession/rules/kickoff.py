"""Rule 1: before the kickoff the ball is dead."""


def find_kickoff(ctx, kick_tag):
    """The kickoff frame: the first frame after Smart Tagging's kickoff tag with the ball seen
    moving faster than kick_speed (within kick_search); the tag plus tag_kickoff_lag if no kick is
    found; the first frame of the half if there is no tag. Never before the half."""
    p = ctx.params
    kick = None
    if kick_tag is not None:
        kick = ctx.first_kick(max(kick_tag, ctx.first), p.kick_search)
        if kick is None:
            kick = kick_tag + p.tag_kickoff_lag
    return max(kick if kick is not None else ctx.first, ctx.first)
