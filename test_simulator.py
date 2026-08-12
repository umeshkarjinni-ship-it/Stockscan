from backtest.simulator import TradeSimulator

print("Starting simulator...")

sim = TradeSimulator(holding_period=20)

sim.run()

print("Finished!")