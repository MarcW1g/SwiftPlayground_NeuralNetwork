//: Playground - noun: a place where people can play

import UIKit

class NeuralKit {
    func sigmoid(_ X:[[Double]]) -> [[Double]]{
        var newArray: [[Double]] = []

        for row in X {
            var newRow: [Double] = []
            for col in row {
                newRow.append(1/(1+exp(-col)))
            }
            newArray.append(newRow)
        }
        return newArray
    }


    func addBias(_ X: [[Double]]) -> [[Double]] {
        var newArray: [[Double]] = []

        for row in X {
            var currentRow = row
            currentRow.insert(1.0, at: 0)
            newArray.append(currentRow)
        }

        return newArray
    }
}

var array = [[0.0,1.0],[1.0,2.0]]
var network = NeuralKit()
var sigmoided = network.sigmoid(array)
print(sigmoided)
var biased = network.addBias(sigmoided)
print(biased)
