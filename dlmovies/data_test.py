import pandas as pd


def main():

    test_data = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3)

    result = [0] * 25000

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv( "data/Bag_of_Words_model_test.csv", index=False, quoting=3 )



if __name__ == '__main__':
    main()