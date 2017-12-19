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
    // MARK: - Neural network
    func sigmoid(_ X:[[Double]]) -> [[Double]] {
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
    func randomWeigts(nrOfInputNodes: Int, nrOfOutputNodes: Int) -> [[Double]] {
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

    func computeLayer(activations: [[Double]], weights: [[Double]]) -> [[Double]]{
        var newArray: [[Double]] = []

        for a_row in activations {
            var temp: [Double] = []
            for w_row in weights {
                temp.append(a_row * w_row)
            }
            newArray.append(temp)
        }
        return newArray
    }

    func oneLayerOut(X: [[Double]], weights: [[Double]]) -> [[Double]] {
        let biasedX = addBias(X)
        return self.sigmoid(computeLayer(activations: biasedX, weights: weights))
    }
}

var network = NeuralKit()
network.oneLayerOut(X: [[1.0,2.0],[4.0,5.0],[7.0,8.0]], weights: [[0.1,0.1,0.2],[0.2,0.2,0.3]])
//network.times(value: 2.0, right: [2.0,3.0])

