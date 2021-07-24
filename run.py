from predicateAssoc import PAssoc
from model import DeepQNetwork
import argparse
import logging

def run(pComb, model):
    step = 0
    for episode in range(200):
        # initial observation
        observation = pComb.reset()
        print(observation)

        while True:
            # find action
            action = model.choose_action(observation)
            # take action and get next observation and reward
            observation_, reward, done = pComb.step(action)

            print(observation_, reward, done, action)

            model.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                model.learn()

            # swap observation
            observation = observation_

            if done:
                break
            step += 1

def main():
    parser = argparse.ArgumentParser(description="Learn Predicate Association")
    parser.add_argument('-d', '--data_dir')
    parser.add_argument('-t', '--threshold', type=float, default=0.2)

    args = parser.parse_args()
    arg_dict = args.__dict__
    for k, v in sorted(arg_dict.items()):
        logging.info('[Argument] %s=%r' % (k, v))
    # run
    pAssoc = PAssoc(arg_dict["threshold"], arg_dict["data_dir"])
    model = DeepQNetwork(pAssoc.getPredicateNum(), pAssoc.getPredicateNum(),
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run(pAssoc, model)


if __name__ == "__main__":
    main()

