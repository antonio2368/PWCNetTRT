//
// Created by Antonio on 07/04/2020.
//

#ifndef PWCNET_PLUGINFACTORY_H
#define PWCNET_PLUGINFACTORY_H

#include "LeakyReluLayer.h"
#include "NvInfer.h"
#include "CostVolumeLayer.h"

#include <unordered_map>
#include <string>
#include <algorithm>
#include <cstdio>


// TODO: RELEASE MEMORY!!!!
class PluginFactory
{
public:
    template< typename T, typename... Args >
    nvinfer1::IPlugin* createPlugin( std::string layerType, Args&&... args )
    {
        static_assert( std::is_base_of< nvinfer1::IPlugin, T >::value, "Template type should be derived from IPlugin" );

        mPlugins.push_back( { layerType, ( nvinfer1::IPlugin* ) new T{ std::forward< Args >( args )... } } );
        return mPlugins.back().second;
    }

    ~PluginFactory()
    {
        for ( auto plugin : mPlugins )
        {
            if ( plugin.first == "costvolume" )
            {
               delete static_cast< CostVolumeLayer* >( plugin.second );
            }
            else if ( plugin.first == "leakyrelu" )
            {
                delete static_cast< LeakyReluLayer* >( plugin.second );
            }
        }
    }

private:
    std::vector< std::pair< std::string, nvinfer1::IPlugin* > > mPlugins;
};


#endif //PWCNET_PLUGINFACTORY_H
