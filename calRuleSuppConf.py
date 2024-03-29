from predicateAssoc import PAssoc
import argparse
import logging


# only for rules containing variable predicates with operator being "="
def main():
    parser = argparse.ArgumentParser(description="Calculate Rule Support and Confidence")
    parser.add_argument('-d', '--data_dir', type=str, default="ncvoter.csv")
    parser.add_argument('-t', '--threshold', type=float, default=0.2)
    parser.add_argument('-c', '--test_file', type=str, default="")
    parser.add_argument('-s', '--ifGenTest', type=bool, default=True)

    args = parser.parse_args()
    arg_dict = args.__dict__
    for k, v in sorted(arg_dict.items()):
        logging.info('[Argument] %s=%r' % (k, v))

    pAssoc = PAssoc(arg_dict["threshold"], arg_dict["data_dir"])
    pAssoc.cal_supp(["party"], "voting_intention", pAssoc.satisfied_tuples)
    pAssoc.cal_supp(["date", "voting_intention"], "election_phase", pAssoc.satisfied_tuples)
    pAssoc.cal_supp(["way_of_voting", "date"], "voting_intention", pAssoc.satisfied_tuples)
    pAssoc.cal_supp(["way_of_voting", "date"], "election_phase", pAssoc.satisfied_tuples)
    pAssoc.cal_supp(["date", "county_desc"], "voting_intention", pAssoc.satisfied_tuples)
    pAssoc.cal_supp(["county_desc"], "county_id", pAssoc.satisfied_tuples)
    pAssoc.cal_supp(["city_id", "county_id"], "city", pAssoc.satisfied_tuples)
    pAssoc.cal_supp(["city"], "county_id", pAssoc.satisfied_tuples)



if __name__ == "__main__":
    main()

