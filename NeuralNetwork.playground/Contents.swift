//: Playground - noun: a place where people can play

import UIKit

extension Array where Iterator.Element == Double {
    static func *(left: [Double], right: [Double]) -> Double {
        var product: Double = 0.0

        for (i, key) in left.enumerated() {
            product += key * right[i]
        }

        return product
    }

    static func *(value: Double, right: [Double]) -> Double {
        var product: Double = 0.0

        for key in right {
            product += value * key
        }

        return product
    }
}

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

    // This function creates a new matrix with doubles between -0.5 and 0.5
    func randomWeigts(nrOfInputNodes: Int, nrOfOutputNodes: Int) -> [[Double]]{
        var newArray: [[Double]] = []
        for _ in 0..<nrOfInputNodes {
            var newRow: [Double] = []
            for _ in 0..<nrOfOutputNodes+1 {
                newRow.append(drand48() - 0.5)
            }
            newArray.append(newRow)
        }
        return newArray
    }
}

var network = NeuralKit()
let newWeigths = network.randomWeigts(nrOfInputNodes: 2, nrOfOutputNodes: 5)
print(newWeigths)

//var array = [[0.0,1.0],[1.0,2.0]]

//var sigmoided = network.sigmoid(array)
//print(sigmoided)
//var biased = network.addBias(sigmoided)
//print(biased)

