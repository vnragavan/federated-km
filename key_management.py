"""
This module defines the KeyManagement class. It generates the cryptocontext object, and consequently generates and stores keys for multiparty computation  
"""
from openfhe import *
from math import log2
import pdb

class KeyManagement:
    def __init__(self, batch_size = 16, mult_depth = 3, scaling_mod_size = 50, sigma = 3.2, dcrtBits = 50, firstMod = 60, 
                 compressionLevel = COMPRESSION_LEVEL.COMPACT, securityLevel = SecurityLevel.HEStd_128_classic):
        """
        Initializes the key generation 
        Parameters:
             Args:        
                mult_depth: int
                Maximum multplicative depth allowed.
                batchSize decides the maxiumum length of a list to used in MakePackedPlaintext and MakeCKKSPackedPlaintext.
                scaling_mod_size 
                sigma 
                dcrtBits
                firstMod = 60, 
                securityLevel,
                compressionLevel: COMPACT or SLACK: SLACK is more secure, but slightly slower, SLACK is the default. 
            See here for an explanation of the various parameters:
        https://github.com/openfheorg/openfhe-development/tree/main/src/pke/examples#description-of-the-cryptocontext-parameters-and-their-restrictions"""
   
        parameters = CCParamsCKKSRNS()
        parameters.SetSecurityLevel(securityLevel)
        parameters.SetStandardDeviation(sigma)
        parameters.SetSecretKeyDist(UNIFORM_TERNARY)
        parameters.SetMultiplicativeDepth(mult_depth)
        parameters.SetKeySwitchTechnique(HYBRID)
        parameters.SetBatchSize(batch_size)
        parameters.SetScalingModSize(dcrtBits)
        parameters.SetFirstModSize(firstMod)
        parameters.SetInteractiveBootCompressionLevel(compressionLevel)
        
        # Generate the crypto context and enable required features.
        self.crypto_context = GenCryptoContext(parameters)
        self.crypto_context.Enable(PKE)
        self.crypto_context.Enable(KEYSWITCH)
        self.crypto_context.Enable(LEVELEDSHE)
        self.crypto_context.Enable(ADVANCEDSHE)
        self.crypto_context.Enable(MULTIPARTY)
        #self.crypto_context.Enable(FHE) # Used for bootstrapping in EvalDivide_test.py
        
         # Output the generated parameters
        print(f"p = {self.crypto_context.GetPlaintextModulus()}")
        print(f"n = {self.crypto_context.GetCyclotomicOrder()/2}")
        print(f"lo2 q = {log2(self.crypto_context.GetModulus())}")

    #for testing
    def generate_keys(self, num_keys: int): #n: int, mult_depth: int, dir = ""
        """ 
        Generates n secret keys, a common public key and common evaluation keys.
        Args:
         num_keys: int
            The number of keys to generate, corresponds to the number of participarnts.
      
        """
        #generate keys
        keys = []
        keys.append (self.crypto_context.KeyGen())
        for i in range(num_keys-1):
            kpi = self.crypto_context.MultipartyKeyGen(keys[i].publicKey)
            keys.append(kpi)
        self.public_key = keys[num_keys-1].publicKey
        self.keys = keys

if __name__ == '__main__':
     #for testing
     batchSize = 16
     print("test key generation")
     key_test = KeyManagement()
     key_test.generate_keys(5)
     print("test key generation")
     vectorOfInts1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
     vectorOfInts2 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     vectorOfInts3 = [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0]
     cc = key_test.crypto_context
     publicKey = key_test.public_key
     keys = key_test.keys
     plaintext1 = cc.MakeCKKSPackedPlaintext(vectorOfInts1)
     plaintext2 = cc.MakeCKKSPackedPlaintext(vectorOfInts2)
     plaintext3 = cc.MakeCKKSPackedPlaintext(vectorOfInts3)
     ciphertext1 = cc.Encrypt(publicKey, plaintext1)
     ciphertext2 = cc.Encrypt(publicKey, plaintext2)
     ciphertext3 = cc.Encrypt(publicKey, plaintext3)


     ciphertextAdd12 = cc.EvalAdd(ciphertext1, ciphertext2)
     ciphertextAdd123 = cc.EvalAdd(ciphertextAdd12, ciphertext3)

     ciphertextPartial1 = cc.MultipartyDecryptLead([ciphertextAdd123], keys[0].secretKey)
     ciphertextPartial2 = cc.MultipartyDecryptMain([ciphertextAdd123], keys[1].secretKey)
     ciphertextPartial3 = cc.MultipartyDecryptMain([ciphertextAdd123], keys[2].secretKey)
     ciphertextPartial4 = cc.MultipartyDecryptMain([ciphertextAdd123], keys[3].secretKey)
     ciphertextPartial5 = cc.MultipartyDecryptMain([ciphertextAdd123], keys[4].secretKey)
     partialCiphertextVec = [ciphertextPartial1[0], ciphertextPartial2[0], ciphertextPartial3[0],
                            ciphertextPartial4[0], ciphertextPartial5[0]]
     plaintextMultipartyNew = cc.MultipartyDecryptFusion(partialCiphertextVec)
    
     print("\n Original Plaintext: \n")
     print(plaintext1)
     print(plaintext2)
     print(plaintext3)

     plaintextMultipartyNew.SetLength(plaintext1.GetLength())

     print("\n Resulting Fused Plaintext: \n")
     print(plaintextMultipartyNew)

     print("\n")
