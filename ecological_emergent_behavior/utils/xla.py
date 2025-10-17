import os

def apply_xla_flags():
    os.environ['XLA_FLAGS'] = '--xla_disable_hlo_passes=rematerialization'
