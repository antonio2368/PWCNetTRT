//
// Created by Antonio on 07/04/2020.
//

#ifndef PWCNET_PLUGINFACTORY_H
#define PWCNET_PLUGINFACTORY_H

#include "LeakyReluLayer.h"
#include "NvInfer.h"

#include <unordered_map>
#include <string>
#include <algorithm>
#include <cstdio>

template< typename T >
class PluginFactory
{
    static_assert( std::is_base_of< nvinfer1::IPlugin, T >::value, "Template type should be derived from IPlugin" );

public:
    template< typename... Args >
    nvinfer1::IPlugin& createPlugin( const char *layerName, Args&&... args )
    {
        std::string strName{ layerName };

        mPlugins.emplace( layerName, std::forward< Args >( args )... );
        return mPlugins.at( layerName );

    }
private:
    std::unordered_map< std::string, T > mPlugins;
};


#endif //PWCNET_PLUGINFACTORY_H
