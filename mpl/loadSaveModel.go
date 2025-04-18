package mpl

import "os"

func (net Network) Save() {
	h, err := os.Create("./trained/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddensWeights.MarshalBinaryTo(h)
	}
	o, err := os.Create("./trained/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputsWeights.MarshalBinaryTo(o)
	}
}

func (net *Network) Load() {
	h, err := os.Open("./trained/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddensWeights.Reset()
		net.hiddensWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("./trained/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputsWeights.Reset()
		net.outputsWeights.UnmarshalBinaryFrom(o)
	}
	return
}
