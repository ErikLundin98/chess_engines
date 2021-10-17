#ifndef NETWORK_H
#define NETWORK_H




struct Network {
    struct Evaluation {
        double value;
        vector action_logits;
    };
    
    Evaluation evaluate(state){
        // Do some torch stuff
    }
};





#endif // NETWORK_H