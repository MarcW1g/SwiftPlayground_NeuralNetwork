//: Playground - noun: a place where people can play

import UIKit

class NeuralKit {
    func sigmoid(X:[[Double]]) -> [[Double]]{
        var newArray: [[Double]] = []

        for item in X {
            var newRow: [Double] = []
            for col in item {
                newRow.append(1/(1+exp(-col)))
            }
            newArray.append(newRow)
        }
        return newArray
    }

//
//    func addBias([[Double]]) -> [[Double]] {
//
//    }
}



//var array = [[0.0,1.0],[1.0,2.0]]
//var network = NeuralKit()
//var sigmoided = network.sigmoid(X: array)
//print(sigmoided)

