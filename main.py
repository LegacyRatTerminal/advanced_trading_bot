# main.py
import asyncio
import signal
import sys
from config.settings import TradingConfig
from src.bot import AdvancedTradingBot

class GracefulExit(SystemExit):
    code = 1

async def main():
    config = TradingConfig()
    bot = AdvancedTradingBot(config)
    
    try:
        await bot.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def handle_exit():
    """
    Synchronous signal handler for graceful shutdown
    """
    print("\nReceived exit signal. Shutting down gracefully...")
    sys.exit(0)

if __name__ == '__main__':
    try:
        # Set up signal handling for Windows and Unix-like systems
        if sys.platform == 'win32':
            # For Windows, use a simple keyboard interrupt handler
            try:
                asyncio.run(main())
            except KeyboardInterrupt:
                handle_exit()
        else:
            # For Unix-like systems, use more advanced signal handling
            loop = asyncio.get_event_loop()
            
            # Add signal handlers
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig, 
                    lambda s=sig: asyncio.create_task(shutdown(s))
                )
            
            asyncio.run(main())
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

async def shutdown(sig):
    """
    Graceful shutdown coroutine
    """
    print(f"\nReceived exit signal {sig.name}...")
    
    # Gather all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    # Cancel all tasks
    for task in tasks:
        task.cancel()
    
    # Wait for tasks to be canceled
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop the event loop
    asyncio.get_event_loop().stop()