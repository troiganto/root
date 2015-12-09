// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSynapse                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// Synapse class used by TMVA artificial neural network methods
//_______________________________________________________________________

#include "TMVA/TSynapse.h"

#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

static const Int_t fgUNINITIALIZED = -1;

ClassImp(TMVA::TSynapse);

TMVA::MsgLogger* TMVA::TSynapse::fgLogger = 0;

//______________________________________________________________________________
TMVA::TSynapse::TSynapse()
   : fWeight( 0 ),
     fLearnRate( 0 ),
     fDelta( 0 ),
     fPrevDelta( 0 ),
     fDEDw( 0 ),
     fCount( 0 ),
     fPreNeuron( NULL ),
     fPostNeuron( NULL )
{
   // constructor
   fWeight     = fgUNINITIALIZED;
   if (!fgLogger) fgLogger = new MsgLogger("TSynapse");
}


//______________________________________________________________________________
TMVA::TSynapse::~TSynapse()
{
   // destructor
}

//______________________________________________________________________________
void TMVA::TSynapse::SetWeight(Double_t weight)
{
   // set synapse weight
   fWeight = weight;
}

//______________________________________________________________________________
Double_t TMVA::TSynapse::GetWeightedValue()
{
   // get output of pre-neuron weighted by synapse weight
   if (fPreNeuron == NULL)
      Log() << kFATAL << "<GetWeightedValue> synapse not connected to neuron" << Endl;

   return (fWeight * fPreNeuron->GetActivationValue());
}

//______________________________________________________________________________
Double_t TMVA::TSynapse::GetWeightedDelta()
{
   // get error field of post-neuron weighted by synapse weight

   if (fPostNeuron == NULL)
      Log() << kFATAL << "<GetWeightedDelta> synapse not connected to neuron" << Endl;

   return fWeight * fPostNeuron->GetDelta();
}

//______________________________________________________________________________
void TMVA::TSynapse::AdjustWeight()
{
   // adjust the weight based on the error field all ready calculated by CalculateDelta
   Double_t wDelta = fDelta / fCount;
   fWeight += -fLearnRate * wDelta;
   InitDelta();
}

//______________________________________________________________________________
void TMVA::TSynapse::CalculateDelta()
{
   // calculate/adjust the error field for this synapse
   fDelta += fPostNeuron->GetDelta() * fPreNeuron->GetActivationValue();
   fCount++;
}

//______________________________________________________________________________
void TMVA::TSynapse::SARPropDecayWeights(Double_t decayFactor)
{
   // SARProp adds a term to the error function that penalizes large weights.
   // E_SARProp(w_ij) = E(w_ij) + 0.5 * k_1 * log(w_ij^2+1) * T
   // w_ij is the current weight,
   // E is the original error function,
   // k_1 is an arbitrary factor, chosen to be 0.01,
   // T is the Temperature, T = 2^(-lambda*i_Epoch) with the cooling speed lambda,
   // i_Epoch is the index of the current epoch.
   //
   // This weight decay term is honored by adding its gradient
   // `k_1 * w_ij/(1+w_ij^2) * 2^(-lambda*i_Epoch)`
   // to the synapse Delta.
   //
   // The parameter `decayFactor` contains the term `k_1 * 2^(-T*i_Epoch)`
   // and has to be passed from outside.

   // Also note that we have to multiply by fCount so that the original
   // Delta and the weight decay term have the same order of magnitude.
   fDelta += fCount * decayFactor * fWeight/(1+fWeight*fWeight);
}

//______________________________________________________________________________
void TMVA::TSynapse::SARPropAdjustWeight()
{
   // Adjust weights according to delta, but only use its sign, not the
   // magnitude.
   if      (fDelta > 0.0) { fWeight -= fLearnRate; }
   else if (fDelta < 0.0) { fWeight += fLearnRate; }
   else                   { }
   if      (fWeight >  150.) { fWeight =  150.; }
   else if (fWeight < -150.) { fWeight = -150.; }
   fPrevDelta = fDelta;
   InitDelta();
}
