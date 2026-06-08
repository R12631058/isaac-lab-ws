import omni.usd; stage = omni.usd.get_context().get_stage(); [print(p.GetPath()) for p in stage.Traverse() if 'UM' in p.GetPath().pathString]
