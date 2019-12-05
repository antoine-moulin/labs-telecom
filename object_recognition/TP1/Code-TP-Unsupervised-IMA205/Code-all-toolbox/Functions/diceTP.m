function [ d ] = diceTP( segIm, grndTruth )
%DICE(I,J) Computes the dice index between binary images of same size

    d = 2*nnz(segIm & grndTruth)/(nnz(segIm) + nnz(grndTruth));

end
