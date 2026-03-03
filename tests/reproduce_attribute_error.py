from pettingzoo.classic import tictactoe_v3
from utils.wrappers import RecordVideo


def reproduce():
    env = tictactoe_v3.env()
    # Wrap with RecordVideo manually
    env = RecordVideo(env, video_folder="reproduce_videos")

    print("Resetting environment...")
    env.reset()

    print("Attempting to call env.last()...")
    try:
        obs, reward, term, trunc, info = env.last()
        print("Success!")
    except AttributeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    reproduce()
