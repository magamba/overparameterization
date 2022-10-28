# -*- coding: utf-8 -*-

""" Data sampling strategies
"""

import logging
from functools import partial
from core.sampling import create_sampler

logger = logging.getLogger(__name__)

def load_sampler(cmd_args):
    """ Initialize MC sampling strategy
    """
    logger.debug("Initializing strategy: {}".format(cmd_args.gen_strategy))
    try:
        strategy = GEN_STRATEGIES[cmd_args.gen_strategy]
    except KeyError:
        raise ValueError("Unrecognized strategy: {}".format(cmd_args.gen_strategy))
        
    return strategy(cmd_args=cmd_args)


GEN_STRATEGIES = {
    "none" : partial(create_sampler, "none"), # default strategy
}

