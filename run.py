from predicateAssoc import PAssoc
from model import DeepQNetwork
import argparse
import logging
import numpy as np

MAX_LHS_PREDICATES = 5

def run(pComb, model):
    step = 0
    for episode in range(20):
        # initial observation
        pComb.reset()
        observation = pComb.initialAction(episode * 777)

        print(observation)

        while True:
            # find action
            action = model.choose_action(observation)
            if action == -1:
                break
            # take action and get next observation and reward
            observation_, reward, done = pComb.step(action)

            print("Epoch {} : {}, {}, {}, {}".format(episode, observation_, reward, done, action))

            model.store_transition(observation, action, reward, observation_)

            if (step > 2) and (step % 2 == 0):
                model.learn()

            # swap observation
            observation = observation_
            if len(observation[observation==1]) > MAX_LHS_PREDICATES:
                break
            if done:
                break
            step += 1

def test(pComb, model, maxLHS, numREEs, ifGenTest, test_file):
    if ifGenTest:
        data = pComb.test_np(maxLHS, numREEs)
        np.save(test_file, data)
    else:
        data = np.load(test_file)
    acc = []
    for record in data:
        predict = model.predictCorrelation(record[:-2], record[-2])
        acc.append(predict == record[-1])
    return acc



def main():
    parser = argparse.ArgumentParser(description="Learn Predicate Association")
    parser.add_argument('-d', '--data_dir', type=str, default="adult_data.csv")
    parser.add_argument('-t', '--threshold', type=float, default=0.2)
    parser.add_argument('-c', '--test_file', type=str, default="")
    parser.add_argument('-s', '--ifGenTest', type=bool, default=True)

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
    maxLHS = 5
    numREEs = 20
    print(test(pAssoc, model, maxLHS, numREEs, arg_dict['ifGenTest'], arg_dict['test_file']))


if __name__ == "__main__":
    main()

